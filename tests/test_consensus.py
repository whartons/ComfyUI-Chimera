import pytest
from scripts.agent.judge import ConsensusJudge, Verdict


class FakeJudge:
    def __init__(self, verdict):
        self._v = verdict
    def judge(self, image_path, rubric):
        return self._v


def _v(passed, score, issues=()):
    return Verdict(passed=passed, score=score, issues=list(issues))


def test_consensus_majority_pass_and_mean_score():
    cj = ConsensusJudge([_FJ(True, 0.9), _FJ(True, 0.8), _FJ(False, 0.4, ["x not met"])])
    out = cj.judge("img.png", None)
    assert out.passed is True                              # 2/3 pass >= threshold 2
    assert abs(out.score - (0.9 + 0.8 + 0.4) / 3) < 1e-6   # mean of sub-scores
    assert out.issues == ["x not met"]                     # surfaced from the dissenter


def test_consensus_minority_fails_unions_issues():
    cj = ConsensusJudge([_FJ(True, 0.9), _FJ(False, 0.3, ["a not met"]), _FJ(False, 0.2, ["b not met"])])
    out = cj.judge("i", None)
    assert out.passed is False                             # only 1/3 pass
    assert out.issues == ["a not met", "b not met"]        # order-preserving union


def test_consensus_dedups_shared_issues():
    cj = ConsensusJudge([_FJ(False, 0.1, ["dup", "x not met"]), _FJ(False, 0.2, ["dup", "y not met"])])
    assert cj.judge("i", None).issues == ["dup", "x not met", "y not met"]   # 'dup' once


def test_consensus_n2_requires_unanimous():
    assert ConsensusJudge([_FJ(True, 0.9), _FJ(True, 0.9)]).judge("i", None).passed is True
    assert ConsensusJudge([_FJ(True, 0.9), _FJ(False, 0.4)]).judge("i", None).passed is False


def test_consensus_custom_threshold():
    cj = ConsensusJudge([_FJ(True, 0.9), _FJ(False, 0.1), _FJ(False, 0.1)], pass_threshold=1)
    assert cj.judge("i", None).passed is True              # threshold=1: any pass -> pass


def test_consensus_flaky_judge_counts_as_fail_not_crash():
    class Boom:
        def judge(self, image_path, rubric):
            raise RuntimeError("kaboom")
    out = ConsensusJudge([_FJ(True, 0.9), _FJ(True, 0.9), Boom()]).judge("i", None)
    assert out.passed is True                              # 2/3 pass; boom counts as a fail
    assert any("judge error" in i for i in out.issues)     # the error is surfaced, not swallowed


def test_consensus_empty_raises():
    with pytest.raises(ValueError):
        ConsensusJudge([])


def _FJ(passed, score, issues=()):
    return FakeJudge(_v(passed, score, issues))
