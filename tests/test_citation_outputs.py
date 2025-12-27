from tiangong_ai_workspace.agents.citation_agent import doi_to_filename


def test_doi_to_filename_sanitizes_slashes() -> None:
    assert doi_to_filename("10.1016/j.resconrec.2020.104917") == "10.1016_j.resconrec.2020.104917.md"


def test_doi_to_filename_strips_doi_org_prefix() -> None:
    assert doi_to_filename("https://doi.org/10.1016/j.resconrec.2020.104917") == "10.1016_j.resconrec.2020.104917.md"
