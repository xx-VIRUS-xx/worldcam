import pytest

from worldcam.graph.config import Neo4jConfig, find_env_file, parse_env_file
from worldcam.graph.schema import apply_prefix, check_prefix, schema_statements


def test_parse_env_file(tmp_path):
    f = tmp_path / ".env"
    f.write_text("# c\nexport NEO4J_URI=bolt://h:1\nNEO4J_PASSWORD='s3 #cret'\n"
                 "NEO4J_USER=bob   # inline\nEMPTY=\nnot a line\n")
    assert parse_env_file(f) == {"NEO4J_URI": "bolt://h:1", "NEO4J_PASSWORD": "s3 #cret",
                                 "NEO4J_USER": "bob", "EMPTY": ""}


def test_config_env_overrides_file_and_redacts(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("NEO4J_PASSWORD=fromfile\nNEO4J_USER=fileuser\n")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    assert find_env_file(sub) == tmp_path / ".env"
    monkeypatch.setenv("NEO4J_USER", "envuser")
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    monkeypatch.delenv("NEO4J_URI", raising=False)
    cfg = Neo4jConfig.from_env(tmp_path / ".env")
    assert (cfg.user, cfg.password, cfg.uri) == ("envuser", "fromfile", "bolt://127.0.0.1:7687")
    assert "fromfile" not in repr(cfg) and "fromfile" not in str(cfg)


def test_prefix_rewrites_labels_only():
    q = "MATCH (f:Face)-[:IN]->(c:FaceCluster)-[:BELONGS_TO]->(p:Person) RETURN f"
    assert apply_prefix(q, "T") == \
        "MATCH (f:TFace)-[:IN]->(c:TFaceCluster)-[:BELONGS_TO]->(p:TPerson) RETURN f"
    assert apply_prefix(q, "") == q
    with pytest.raises(ValueError):
        check_prefix("x`) DETACH DELETE n //")
    stmts = schema_statements()
    assert any("VECTOR INDEX face_embedding" in s and "512" in s and "'cosine'" in s for s in stmts)


@pytest.mark.parametrize("q", [
    "CREATE (n)", "match (n) detach delete n", "MATCH (n) SET n.x = 1", "MATCH (n) REMOVE n.x",
    "MERGE (n:Person {id: 'a'})", "CALL apoc.periodic.iterate('a','b',{})", "CALL dbms.killQuery('x')",
    "MATCH (n) RETURN n; MATCH (m) DELETE m", "LOAD CSV FROM 'x' AS r RETURN r",
    "MATCH (n) CALL (n) { SET n.x = 1 } IN TRANSACTIONS RETURN n", "SHOW USERS", "USE system SHOW USERS",
    "MATCH (n) FOREACH (x IN [1] | SET n.y = x)", "DROP INDEX person_id",
])
def test_check_read_only_rejects(q):
    from worldcam.graph.store import check_read_only
    with pytest.raises(ValueError):
        check_read_only(q)


@pytest.mark.parametrize("q", [
    "MATCH (p:Person)-[:RELATED]->(o) WHERE p.name CONTAINS $t RETURN p.name, o.offset LIMIT 5",
    "CALL db.index.vector.queryNodes('face_embedding', 3, $v) YIELD node, score RETURN node.id, score",
    "MATCH (p:Person) CALL (p) { MATCH (p)-[:HAS_FACT]->(f) RETURN count(f) AS n } RETURN p.id, n",
])
def test_check_read_only_allows(q):
    from worldcam.graph.store import check_read_only
    check_read_only(q)
