# Copyright 2026 pytest-adk contributors
"""Pytest-friendly ADK evaluation helpers."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

try:
  import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
  import tomli as tomllib

from google.adk.evaluation import AgentEvaluator as _AdkAgentEvaluator
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_config import get_eval_metrics_from_config
from google.adk.evaluation.eval_result import EvalCaseResult
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.local_eval_set_results_manager import (
    LocalEvalSetResultsManager,
)
from google.adk.evaluation.simulation.user_simulator_provider import (
    UserSimulatorProvider,
)

from .prompt_template import _expand_prompt_templates

logger = logging.getLogger(__name__)

_EVAL_APP_NAME = 'test_app'
_NUM_RUNS = 2

# Subpath that ADK's LocalEvalSetResultsManager writes ``*.evalset_result.json``
# files into, relative to ``{results_dir}/{app_name}/``.
_ADK_EVAL_HISTORY_SUBDIR = Path('.adk') / 'eval_history'

_RESULT_FILE_SUFFIX = '.evalset_result.json'
_UNIX_TIMESTAMP_SUFFIX_RE = re.compile(r'_\d+\.\d+$')
_LOCAL_DATETIME_FORMAT = '%Y%m%d-%H%M%S'
# Hidden and suffix-less, so neither ADK's listdir-based discovery nor the
# suffix glob below can ever see files that are still being staged.
_STAGING_PREFIX = '.staging-'
# A staging directory quiet for this long is considered abandoned by a killed
# process and gets recovered; younger ones may belong to a save in flight.
_STAGING_RECOVERY_AGE_SECONDS = 3600
# Working-copy names inside staging. Suffix-less, so they can never be taken
# for results. The publishable copy is what publication hard-links into
# history: its link count doubles as crash-safe proof that publication
# happened, which recovery consults before touching the staged original.
_PUBLISHABLE_NAME = '.publishable.tmp'
_ALIGNING_NAME = '.aligning.tmp'


def _dump_result_document(payload: dict) -> str:
  r"""Serializes a result document the way ADK itself writes one.

  ``ensure_ascii=False`` keeps multibyte text -- Japanese prompts and
  responses, ... -- readable: ADK writes results through pydantic's
  ``model_dump_json``, which emits them as UTF-8, so republishing the
  document under a readable name must not turn them into ``\uXXXX``
  escapes on the way back out.

  A document whose text has no UTF-8 form at all -- ``json.loads`` accepts
  an escaped lone surrogate such as ``\ud800`` -- falls back to the escaped
  form. Writing it would otherwise raise ``UnicodeEncodeError``, which no
  caller expects: it would abort a salvage pass mid-way and strand the
  results it had not moved yet.
  """
  readable = json.dumps(payload, indent=2, ensure_ascii=False)
  try:
    readable.encode('utf-8')
  except UnicodeEncodeError:
    return json.dumps(payload, indent=2)
  return readable


class _ReadableNameEvalSetResultsManager(LocalEvalSetResultsManager):
  """LocalEvalSetResultsManager that names files by local datetime.

  ADK names each result file ``{app}_{eval_set}_{time.time()}`` (a bare unix
  float). This publishes the file as local ``YYYYMMDD-HHMMSS`` instead,
  keeping ADK's ``.evalset_result.json`` suffix so ``list_eval_set_results``
  / ``get_eval_set_result`` / ``adk web`` still discover it.

  Each save lets ADK write into a private staging directory, so this
  process's file is identified directly rather than by diffing the shared
  history directory (concurrent savers cannot confuse each other's output).
  The embedded ids are rewritten to the readable stem while the file is
  still unpublished, then the document is published atomically with
  ``os.link`` -- a concurrent same-second save gets ``FileExistsError`` and
  takes a ``-2``, ``-3``, ... counter instead of overwriting, and no empty,
  partial, or id-mismatched file ever appears under a discoverable name.
  If any step of the readable-name path fails, the file ADK wrote is
  published unchanged under ADK's own unix-timestamp name, keeping filename
  and embedded ids consistent on every path.
  """

  def __init__(self, agents_dir: str) -> None:
    super().__init__(agents_dir=agents_dir)
    # Own attribute rather than the parent's private ``_agents_dir``.
    self._results_root = agents_dir

  def save_eval_set_result(
      self,
      app_name: str,
      eval_set_id: str,
      eval_case_results: list[EvalCaseResult],
  ) -> None:
    """Saves via ADK into staging, then publishes under a datetime stem.

    Args:
        app_name: ADK app name; also the results subdirectory.
        eval_set_id: The evalset's id, embedded in the saved filename.
        eval_case_results: Results to persist, forwarded to ADK unchanged.
    """
    history_dir = Path(self._results_root) / app_name / _ADK_EVAL_HISTORY_SUBDIR
    history_dir.mkdir(parents=True, exist_ok=True)
    # Give results stranded by an earlier killed process a way back into
    # history before this save creates its own staging directory.
    self._recover_abandoned_staging(history_dir)
    # Inside history_dir so the hard link below stays on one filesystem.
    staging_dir = Path(tempfile.mkdtemp(prefix=_STAGING_PREFIX, dir=history_dir))
    try:
      LocalEvalSetResultsManager(
          agents_dir=os.fspath(staging_dir)
      ).save_eval_set_result(
          app_name=app_name,
          eval_set_id=eval_set_id,
          eval_case_results=eval_case_results,
      )
      staged = list(
          (staging_dir / app_name / _ADK_EVAL_HISTORY_SUBDIR).glob(
              '*' + _RESULT_FILE_SUFFIX
          )
      )
      if len(staged) == 1:
        try:
          self._publish_readable(staged[0], history_dir)
          # Published under the readable name; drop the staged original so
          # the salvage below does not publish it a second time. The
          # working copy goes second: until the original is gone, its
          # link count remains the physical proof of publication that
          # recovery consults if this process dies mid-cleanup.
          staged[0].unlink()
          staged[0].with_name(_PUBLISHABLE_NAME).unlink(missing_ok=True)
        except Exception:  # noqa: BLE001 - salvage publishes it as-is
          pass
    finally:
      self._salvage_staging(staging_dir)

  def _recover_abandoned_staging(self, history_dir: Path) -> None:
    """Publishes results orphaned in staging by a killed process.

    A save interrupted after ADK wrote the result but before publication
    leaves the completed document hidden in its private ``.staging-*``
    directory, where no later run would otherwise look. Directories whose
    entire tree has been quiet for ``_STAGING_RECOVERY_AGE_SECONDS`` are
    atomically renamed -- claiming them, so concurrent recoverers cannot
    process the same one -- and salvaged like this save's own staging.
    Younger directories are left alone: they may belong to a save still in
    flight, and salvaging mid-write could publish a partial document. The
    claimed name is a fresh fixed-length one (never a suffix appended to
    the old name, which would grow on every failed recovery until rename
    hits the filesystem's name limit and the directory becomes permanently
    unrecoverable); it keeps the staging prefix and its quiet clock is
    restarted, so a recovery that itself dies is simply picked up again by
    a later save.
    """
    now = time.time()
    for entry in list(history_dir.glob(_STAGING_PREFIX + '*')):
      try:
        if not entry.is_dir():
          continue
        newest = max(p.stat().st_mtime for p in (entry, *entry.rglob('*')))
        if now - newest < _STAGING_RECOVERY_AGE_SECONDS:
          continue
        claimed = entry.with_name(
            f'{_STAGING_PREFIX}recovering-{os.urandom(4).hex()}'
        )
        entry.rename(claimed)
        # Restart the quiet clock: concurrent savers leave the claimed
        # directory alone while this salvage runs, and if this process dies
        # the age check makes it eligible again an hour later.
        os.utime(claimed)
      except OSError:
        # Recovered, removed, or still being written by someone else.
        continue
      with contextlib.suppress(OSError):
        self._salvage_staging(claimed)

  def _salvage_staging(self, staging_dir: Path) -> None:
    """Moves everything left in ``staging_dir`` into the real tree.

    The staging directory mirrors the ``agents_dir`` layout, so each
    remaining file is relocated to the same relative path under the real
    results root -- whatever ADK named it. Deleting unrecognized content
    instead would silently drop results the moment ADK changes its output
    layout; this way a save always lands on disk, at worst under ADK's own
    names (whose embedded ids already match). The (by then empty) staging
    tree is removed only after every file has been moved out, so nothing is
    ever deleted with data still inside.
    """
    for path in sorted(staging_dir.rglob('*')):
      if not path.is_file():
        continue
      if path.name in (_PUBLISHABLE_NAME, _ALIGNING_NAME):
        # Working copies, not results. The publishable one doubles as the
        # publication proof consulted below; the staged original remains
        # authoritative for both.
        continue
      destination = Path(self._results_root) / path.relative_to(staging_dir)
      if destination.name.endswith(_RESULT_FILE_SUFFIX):
        if self._publication_proven(path.with_name(_PUBLISHABLE_NAME)):
          # Physical proof, not a content heuristic: publication hard-links
          # the working copy to its final name, so a link count above one
          # means this directory's save already reached history. Skipping
          # the original is then deduplication of the very same save. A
          # content comparison could never establish this -- a distinct
          # run with identical output must be salvaged, and without proof
          # it is.
          continue
        payload = self._load_result_document(path)
        if payload is None:
          # A killed ADK write can leave a truncated document, and age
          # alone is no proof of completeness. Publish it quarantined:
          # preserved for inspection, but invisible to suffix-based
          # discovery, so ADK never serves a corrupt result.
          destination = destination.with_name(destination.name + '.invalid')
      destination.parent.mkdir(parents=True, exist_ok=True)
      self._move_without_overwrite(path, destination)
    shutil.rmtree(staging_dir, ignore_errors=True)

  @staticmethod
  def _publication_proven(publishable: Path) -> bool:
    """Whether this staging directory's save provably reached history.

    Publication hard-links the rewritten working copy to its final name,
    so a working copy whose link count exceeds one is physical evidence
    that the published file exists in history. This is the only condition
    under which salvage skips a result document.
    """
    try:
      return publishable.stat().st_nlink >= 2
    except OSError:
      return False

  @staticmethod
  def _load_result_document(path: Path) -> dict | None:
    """Returns the parsed result document, or None when it is not one."""
    try:
      payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
      return None
    return payload if isinstance(payload, dict) else None

  @staticmethod
  def _numbered(destination: Path, counter: int) -> Path:
    """Returns the ``counter``-th candidate name for ``destination``.

    Numbering goes before ``.evalset_result.json`` (or before a plain last
    suffix) so a deduplicated result file is still discoverable.
    """
    if counter == 1:
      return destination
    name = destination.name
    if name.endswith(_RESULT_FILE_SUFFIX):
      base = name.removesuffix(_RESULT_FILE_SUFFIX)
      return destination.with_name(f'{base}-{counter}{_RESULT_FILE_SUFFIX}')
    return destination.with_name(
        f'{destination.stem}-{counter}{destination.suffix}'
    )

  @classmethod
  def _move_without_overwrite(cls, path: Path, destination: Path) -> None:
    """Moves ``path`` to ``destination``, deduplicating instead of replacing.

    A taken name gets a ``-2``, ``-3``, ... variant rather than being
    overwritten or -- worse -- the move being skipped, which would let the
    caller's cleanup delete ``path`` with this save's data still inside.
    ``os.link`` claims the name and publishes the complete document in one
    atomic step. Where hard links are unsupported, a hidden ``.lock`` name
    is claimed instead of the destination itself, so even a crash mid-move
    can never leave an empty file under a discoverable result name; the
    data then lands via atomic ``os.replace``.
    """
    counter = 1
    while True:
      candidate = cls._numbered(destination, counter)
      # Still in staging here, so the rewrite is invisible until the
      # atomic claim below publishes the finished document.
      cls._align_embedded_ids(path, candidate)
      try:
        os.link(path, candidate)
      except FileExistsError:
        counter += 1
        continue
      except OSError:
        # No hard-link support (e.g. FAT/exFAT). The exclusive-create claim
        # goes to a hidden lock name -- never to ``candidate`` itself, which
        # a crash would leave behind as an empty result file.
        lock = candidate.with_name(f'.{candidate.name}.lock')
        try:
          with open(lock, 'x', encoding='utf-8'):
            pass
        except FileExistsError:
          counter += 1
          continue
        if candidate.exists():
          # Claimed a name someone already published to (they used os.link,
          # or predate the lock protocol). Nothing was or will be replaced
          # here, so releasing the lock is safe: any later claimer re-runs
          # this same exists() check and also skips.
          lock.unlink(missing_ok=True)
          counter += 1
          continue
        os.replace(path, candidate)
        # The lock is deliberately kept as a permanent, one-shot claim on
        # this published name. Releasing it would let a later mover claim
        # the same lock and reach its own exists()/replace with this file
        # already present -- reopening the overwrite race the lock exists
        # to close. Locks accrue only on the no-hard-link salvage path,
        # and their names are invisible to ADK's suffix-based discovery.
        return
      path.unlink()
      return

  @staticmethod
  def _align_embedded_ids(path: Path, candidate: Path) -> None:
    """Aligns a recognized result document's embedded ids with ``candidate``.

    Deduplicating to a ``-2`` name without touching the content would leave
    the file identifying itself as the preexisting one, so a lookup by the
    listed stem could return the wrong result. Only documents that are
    recognizably eval set results (result suffix, JSON object carrying
    ``eval_set_result_id``) are rewritten; anything else is moved untouched
    -- preserving the data takes priority over aligning it.
    """
    if not candidate.name.endswith(_RESULT_FILE_SUFFIX):
      return
    try:
      payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
      return
    if not isinstance(payload, dict) or 'eval_set_result_id' not in payload:
      return
    stem = candidate.name.removesuffix(_RESULT_FILE_SUFFIX)
    if (
        payload.get('eval_set_result_id') == stem
        and payload.get('eval_set_result_name') == stem
    ):
      return
    payload['eval_set_result_id'] = stem
    payload['eval_set_result_name'] = stem
    # ``path`` is the only copy of this result, so it is never written in
    # place: the aligned document goes to a staging-local sibling and lands
    # via atomic replace. If that fails (disk full, ...), the original moves
    # on unaligned rather than risking corruption.
    aligned = path.with_name(_ALIGNING_NAME)
    try:
      aligned.write_text(_dump_result_document(payload), encoding='utf-8')
      os.replace(aligned, path)
    except OSError:
      with contextlib.suppress(OSError):
        aligned.unlink()

  @staticmethod
  def _publish_readable(saved: Path, history_dir: Path) -> None:
    """Publishes staged file ``saved`` under a local-datetime name.

    Raises:
        Exception: When any step fails; ``saved`` is left untouched so the
            caller can publish it under ADK's original name instead.
    """
    base, replaced = _UNIX_TIMESTAMP_SUFFIX_RE.subn(
        '', saved.name.removesuffix(_RESULT_FILE_SUFFIX)
    )
    if replaced != 1:
      raise ValueError(f'unexpected ADK result file name: {saved.name}')
    payload = json.loads(saved.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
      raise ValueError('unexpected ADK result payload shape')
    # The rewritten document lives next to ``saved`` in staging, keeping
    # ``saved`` pristine for the caller's fail-soft path. Its name avoids the
    # result-file suffix so the salvage pass can never mistake this working
    # copy for a result. It is deliberately NOT removed here: after a
    # successful link its link count is the crash-safe proof of publication
    # (the caller removes it only after the staged original is gone), and on
    # failure it carries no unique data and is swept out with the staging
    # tree.
    publishable = saved.with_name(_PUBLISHABLE_NAME)
    stamp = datetime.now().strftime(_LOCAL_DATETIME_FORMAT)
    counter = 1
    while True:
      numbering = '' if counter == 1 else f'-{counter}'
      target = history_dir / f'{base}_{stamp}{numbering}{_RESULT_FILE_SUFFIX}'
      # Keep the embedded id in step with the name about to be claimed,
      # before anything is published. creation_timestamp is deliberately
      # left as the raw unix float: adk web compares it to find the most
      # recent run for an eval set.
      new_stem = target.name.removesuffix(_RESULT_FILE_SUFFIX)
      payload['eval_set_result_id'] = new_stem
      payload['eval_set_result_name'] = new_stem
      publishable.write_text(_dump_result_document(payload), encoding='utf-8')
      try:
        # link() atomically claims the name -- a concurrent save (another
        # process writing the same app/eval set in the same second) gets
        # FileExistsError and moves on to the next counter instead of
        # silently overwriting this result -- and publishes the complete
        # rewritten document in the same step, so no empty or partial file
        # ever appears under a ``*.evalset_result.json`` name, even if
        # this process dies mid-save.
        os.link(publishable, target)
      except FileExistsError:
        counter += 1
        continue
      return


def _load_eval_set_from_toml(eval_set_file: str | Path) -> EvalSet:
  """Load an EvalSet from a TOML file (new EvalSet schema only).

  Unlike ADK's JSON loader, this does not support the legacy data format or an
  explicit ``initial_session``; the initial session must be expressed inside the
  EvalSet schema.

  ``tomllib.load`` requires a binary handle, so we read text and use
  ``tomllib.loads`` to stay consistent with the rest of the package.

  Args:
      eval_set_file: Path to a TOML file containing the ADK ``EvalSet`` schema.

  Returns:
      The validated ADK ``EvalSet`` model.
  """
  data = tomllib.loads(Path(eval_set_file).read_text(encoding='utf-8'))
  return EvalSet.model_validate(data)


def _collect_eval_sets(
    eval_dataset_file_path_or_dir: str | Path,
    *,
    prompt_template_engine: str = 'string',
    initial_session_file: str | None = None,
    eval_config_override: EvalConfig | None = None,
) -> list[tuple[EvalSet, EvalConfig]]:
  """Load evalset files into ``(EvalSet, EvalConfig)`` pairs.

  This centralizes the evalset discovery and loading logic (directory walk for
  ``*.test.json``/``*.test.toml``, direct-file loading, TOML vs. JSON
  branching, and ``<prompt:...>`` template expansion) so it can be shared
  between the ``AgentEvaluator`` pytest fixture path
  (:meth:`_AgentEvaluator.evaluate`) and the future ``pytest-adk eval`` CLI,
  which needs the same loading behavior without going through the fixture.

  Args:
      eval_dataset_file_path_or_dir: Evalset file, or a directory searched
          recursively for ``*.test.json`` and ``*.test.toml`` files.
      prompt_template_engine: Engine used to render ``<prompt:...>`` markers,
          either ``'string'`` (default, ``string.Template``) or ``'jinja'``.
      initial_session_file: Optional initial session file for JSON evalsets.
          TOML evalsets reject this because they support the current
          ``EvalSet`` schema only.
      eval_config_override: When given, this config is used for every evalset
          and the sibling ``test_config.json`` discovery is skipped entirely.
          The CLI passes ``--config-file-path`` here: without it, an
          unloadable sibling config would abort the run even though the
          caller supplied an explicit config meant to replace it.

  Returns:
      A list of ``(EvalSet, EvalConfig)`` pairs, one per discovered evalset
      file, in the order the files were found.

  Raises:
      AssertionError: If ``initial_session_file`` is set and a TOML evalset is
          encountered.
  """
  eval_dataset_path = os.fspath(eval_dataset_file_path_or_dir)
  test_files: list[str] = []
  if os.path.isdir(eval_dataset_path):
    # When a directory is given, only the ADK naming convention
    # (``.test.json`` / ``.test.toml``) is picked up recursively. This keeps
    # sibling files such as ``test_config.json`` (eval metrics) and the
    # ``*.evalset_result.json`` files written by this helper from being
    # mistakenly loaded as evalsets.
    for root, _, files in os.walk(eval_dataset_path):
      for file in files:
        if file.endswith(('.test.json', '.test.toml')):
          test_files.append(os.path.join(root, file))
  else:
    # A directly specified file is taken at face value; the user's intent is
    # explicit. Extension routing (and a naming-convention warning) happens
    # per file below.
    test_files = [eval_dataset_path]

  initial_session = _AdkAgentEvaluator._get_initial_session(
      initial_session_file
  )

  eval_sets: list[tuple[EvalSet, EvalConfig]] = []
  for test_file in test_files:
    # Files discovered via a directory always satisfy the convention, so this
    # only fires for directly specified files that skip the ``.test.`` infix.
    # The check uses the basename so a ``.test.`` directory name does not mask
    # a non-conventional file.
    if '.test.' not in os.path.basename(test_file):
      logger.warning(
          'Evalset file %r does not follow the .test.json/.test.toml naming'
          ' convention; loading it anyway because it was specified directly.',
          test_file,
      )

    eval_config = (
        eval_config_override
        if eval_config_override is not None
        else _AdkAgentEvaluator.find_config_for_test_file(test_file)
    )
    if test_file.endswith('.toml'):
      assert len(initial_session) == 0, (
          'Initial session should be specified as a part of the EvalSet file.'
          ' An explicit initial_session_file is not supported for TOML'
          ' evalsets, which use the EvalSet schema only.'
      )
      eval_set = _load_eval_set_from_toml(test_file)
    else:
      eval_set = _AdkAgentEvaluator._load_eval_set_from_file(
          test_file, eval_config, initial_session
      )

    eval_set = _expand_prompt_templates(
        eval_set, Path(test_file).parent, prompt_template_engine
    )
    eval_sets.append((eval_set, eval_config))

  return eval_sets


@contextlib.contextmanager
def _registered_custom_metrics(
    eval_config: EvalConfig, eval_set_id: str
) -> Iterator[None]:
  """Scopes this config's custom metrics to one evalset's evaluation.

  ``AgentEvaluator._get_eval_results_by_eval_id`` builds its
  ``LocalEvalService`` without a ``metric_evaluator_registry``, so unlike
  ``pytest-adk eval`` -- which passes its own -- this path can only reach
  ADK's process-wide default registry, the object ADK's own ``adk eval``
  command registers into too.

  Entered and exited per evalset, which is what keeps that shared registry
  from turning one test's metrics into every later test's. Within the block
  the evalset's own custom metrics win (including over a built-in of the same
  name); on the way out the previous mapping comes back, so evalsets that
  disagree about a metric name -- across two tests, or across two evalsets in
  one ``evaluate()`` call -- are each still scored with the definition their
  own config names.

  The import is deferred rather than made at module scope because this module
  is loaded by the pytest plugin at interpreter start-up, for every pytest run
  in the environment; :mod:`pytest_adk.metrics` in turn only reaches for
  google-adk's eval dependency chain (see its module docstring) when a config
  actually declares custom metrics.

  Raises:
      ValueError: If a configured custom metric cannot be resolved, or if a
          metric named in ``criteria`` has no evaluator.
  """
  from .metrics import registered_custom_metrics

  with registered_custom_metrics(eval_config, eval_set_id=eval_set_id):
    yield


class _AgentEvaluator:
  """ADK AgentEvaluator wrapper that persists local eval results.

  Construct with a ``results_dir``; the ``AgentEvaluator`` pytest fixture binds
  it to pytest's ``tmp_path`` (see :mod:`pytest_adk.plugin`).
  """

  def __init__(
      self,
      results_dir: str | Path,
      prompt_template_engine: str = 'string',
  ) -> None:
    """Create an evaluator that writes ADK eval history under ``results_dir``.

    Args:
        results_dir: Base directory passed to ADK's local eval results manager.
            The pytest fixture supplies pytest's per-test ``tmp_path``.
        prompt_template_engine: Engine used to render ``<prompt:...>`` markers,
            either ``'string'`` (default, ``string.Template``) or ``'jinja'``.
            The pytest fixture supplies the value of the
            ``pytest_adk_prompt_template_engine`` ini option.
    """
    self._results_dir = results_dir
    self._prompt_template_engine = prompt_template_engine

  @property
  def results_dir(self) -> str | Path:
    """Directory under which eval results are saved."""
    return self._results_dir

  @property
  def eval_history_dir(self) -> Path:
    """Directory where ADK writes ``*.evalset_result.json`` files.

    This mirrors the layout used by
    :class:`~google.adk.evaluation.local_eval_set_results_manager.LocalEvalSetResultsManager`
    so the location can be surfaced (e.g. by the ``AgentEvaluator`` plugin's
    terminal summary) without re-deriving the path elsewhere.
    """
    return Path(self._results_dir) / _EVAL_APP_NAME / _ADK_EVAL_HISTORY_SUBDIR

  async def evaluate(
      self,
      agent_module: str,
      eval_dataset_file_path_or_dir: str | Path,
      num_runs: int = _NUM_RUNS,
      agent_name: str | None = None,
      initial_session_file: str | None = None,
      print_detailed_results: bool = True,
  ) -> None:
    """Evaluate an ADK agent and save generated eval results to disk.

    This mirrors :meth:`google.adk.evaluation.AgentEvaluator.evaluate`, with an
    added persistence hook that saves the per-test-file ``EvalSetResult`` under
    the bound ``results_dir`` before metric failures are asserted.

    Example:
        .. code-block:: python

           @pytest.mark.asyncio
           async def test_with_single_test_file(AgentEvaluator):
             await AgentEvaluator.evaluate(
                 agent_module='home_automation_agent',
                 eval_dataset_file_path_or_dir=(
                     'tests/integration/fixture/home_automation_agent/'
                     'simple_test.test.json'
                 ),
             )

        ``AgentEvaluator`` is a pytest fixture (auto-registered via the
        ``pytest11`` entry point) that binds ``results_dir`` to pytest's
        ``tmp_path``. Eval result JSON files are written under
        ``results_dir/test_app/.adk/eval_history/``.

    Background:
        This helper was inspired by the workflow described in:
        https://nikkie-ftnext.hatenablog.com/entry/google-adk-python-evaluation-use-local-eval-set-results-manager
        The upstream ADK PR for optional eval result persistence is still open:
        https://github.com/google/adk-python/pull/4414

    Args:
        agent_module: Import path of the ADK agent module to evaluate.
        eval_dataset_file_path_or_dir: Evalset file, or a directory searched
            recursively for ``*.test.json`` and ``*.test.toml`` files.
        num_runs: Number of ADK evaluation runs per eval case.
        agent_name: Optional agent variable name inside ``agent_module``.
        initial_session_file: Optional initial session file for JSON evalsets.
            TOML evalsets reject this because they support the current
            ``EvalSet`` schema only.
        print_detailed_results: Whether ADK should print detailed metric output.

    Raises:
        AssertionError: If any ADK metric fails, after eval results have been
            saved to disk.
    """
    eval_sets = _collect_eval_sets(
        eval_dataset_file_path_or_dir,
        prompt_template_engine=self._prompt_template_engine,
        initial_session_file=initial_session_file,
    )

    for eval_set, eval_config in eval_sets:
      await self._evaluate_eval_set_and_save(
          agent_module=agent_module,
          eval_set=eval_set,
          eval_config=eval_config,
          num_runs=num_runs,
          agent_name=agent_name,
          print_detailed_results=print_detailed_results,
      )

  async def _evaluate_eval_set_and_save(
      self,
      *,
      agent_module: str,
      eval_set: EvalSet,
      eval_config: EvalConfig,
      num_runs: int,
      agent_name: str | None,
      print_detailed_results: bool,
  ) -> None:
    """Run ADK evaluation for one ``EvalSet``, persist it, then assert metrics.

    The whole body runs inside :func:`_registered_custom_metrics`, whose scope
    ends only once this evalset is completely done with: ADK consults the
    metric registry while *scoring*, inside
    ``_get_eval_results_by_eval_id()``, and the reporting below reads the
    results that scoring produced. Leaving on the assertion path matters as
    much as leaving on the happy one -- a failing evalset must not be the one
    that leaks its metrics into the rest of the session.
    """
    with _registered_custom_metrics(eval_config, eval_set.eval_set_id):
      agent_for_eval = await _AdkAgentEvaluator._get_agent_for_eval(
          module_name=agent_module, agent_name=agent_name
      )
      eval_metrics = get_eval_metrics_from_config(eval_config)
      user_simulator_provider = UserSimulatorProvider(
          user_simulator_config=eval_config.user_simulator_config
      )

      eval_results_by_eval_id = (
          await _AdkAgentEvaluator._get_eval_results_by_eval_id(
              agent_for_eval=agent_for_eval,
              eval_set=eval_set,
              eval_metrics=eval_metrics,
              num_runs=num_runs,
              user_simulator_provider=user_simulator_provider,
          )
      )

      results_manager = _ReadableNameEvalSetResultsManager(
          agents_dir=os.fspath(self._results_dir)
      )
      all_eval_results: list[EvalCaseResult] = [
          result
          for eval_results_per_eval_id in eval_results_by_eval_id.values()
          for result in eval_results_per_eval_id
      ]
      results_manager.save_eval_set_result(
          app_name=_EVAL_APP_NAME,
          eval_set_id=eval_set.eval_set_id,
          eval_case_results=all_eval_results,
      )

      failures: list[str] = []
      for eval_results_per_eval_id in eval_results_by_eval_id.values():
        eval_metric_results = (
            _AdkAgentEvaluator._get_eval_metric_results_with_invocation(
                eval_results_per_eval_id
            )
        )
        failures_per_eval_case = (
            _AdkAgentEvaluator._process_metrics_and_get_failures(
                eval_metric_results=eval_metric_results,
                print_detailed_results=print_detailed_results,
                agent_module=agent_name,
            )
        )
        failures.extend(failures_per_eval_case)

      failure_message = 'Following are all the test failures.'
      if not print_detailed_results:
        failure_message += (
            ' If you looking to get more details on the failures, then please'
            ' re-run this test with `print_detailed_results` set to `True`.'
        )
      failure_message += '\n' + '\n'.join(failures)
      assert not failures, failure_message
