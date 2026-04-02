"""Tests for storage repositories."""
import pytest
from norvel_writer.storage.repositories.project_repo import ProjectRepo
from norvel_writer.storage.repositories.document_repo import DocumentRepo
from norvel_writer.storage.repositories.draft_repo import DraftRepo


def test_create_and_get_project(db):
    repo = ProjectRepo(db)
    pid = repo.create_project("My Novel", description="A story", language="en")
    project = repo.get_project(pid)
    assert project is not None
    assert project["name"] == "My Novel"
    assert project["language"] == "en"


def test_list_projects(db):
    repo = ProjectRepo(db)
    repo.create_project("Project A")
    repo.create_project("Project B")
    projects = repo.list_projects()
    assert len(projects) >= 2


def test_update_project(db):
    repo = ProjectRepo(db)
    pid = repo.create_project("Old Name")
    repo.update_project(pid, name="New Name")
    p = repo.get_project(pid)
    assert p["name"] == "New Name"


def test_delete_project(db):
    repo = ProjectRepo(db)
    pid = repo.create_project("To Delete")
    repo.delete_project(pid)
    assert repo.get_project(pid) is None


def test_create_chapter(db, project_id):
    repo = ProjectRepo(db)
    cid = repo.create_chapter(project_id, "Chapter One")
    ch = repo.get_chapter(cid)
    assert ch is not None
    assert ch["title"] == "Chapter One"
    assert ch["project_id"] == project_id


def test_list_chapters_ordered(db, project_id):
    repo = ProjectRepo(db)
    repo.create_chapter(project_id, "Chapter A", position=1)
    repo.create_chapter(project_id, "Chapter B", position=2)
    chapters = repo.list_chapters(project_id)
    titles = [c["title"] for c in chapters]
    assert "Chapter A" in titles
    assert "Chapter B" in titles
    assert titles.index("Chapter A") < titles.index("Chapter B")


def test_create_document(db, project_id):
    repo = DocumentRepo(db)
    did = repo.create_document(
        project_id=project_id,
        file_path="/tmp/test.txt",
        file_hash="abc123",
        doc_type="codex",
        fmt="txt",
        title="My Codex",
    )
    doc = repo.get_document(did)
    assert doc["title"] == "My Codex"
    assert doc["status"] == "pending"


def test_draft_accept_flow(db, project_id):
    pr = ProjectRepo(db)
    cid = pr.create_chapter(project_id, "Ch 1")
    dr = DraftRepo(db)
    did1 = dr.create_draft(cid, "First draft content", "llama3.2:3b")
    did2 = dr.create_draft(cid, "Second draft content", "llama3.2:3b")
    dr.accept_draft(did2)
    accepted = dr.get_accepted_draft(cid)
    assert accepted["id"] == did2
    assert accepted["content"] == "Second draft content"
