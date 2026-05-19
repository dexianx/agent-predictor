# Knowledge Graph Pipeline

This pipeline builds a small, rerunnable Neo4j knowledge graph and SQLite persona
database for the DSN x BCT review simulation/recommendation tasks.

## Setup

```bash
cd /Users/fadex/Downloads/coding-apps/data-scraping
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login
docker compose up -d neo4j
```

Neo4j Browser will be available at <http://localhost:7474>.

## Run

Small local-only smoke test:

```bash
python -m pipeline.run_pipeline \
  --sample-size 50 \
  --amazon-max-rows 0 \
  --skip-neo4j \
  --use-hash-embedder \
  --sqlite-path /tmp/personas-smoke.db
```

Default 5,000-row graph load:

```bash
python -m pipeline.run_pipeline \
  --sample-size 5000 \
  --reset-neo4j \
  --sqlite-path personas.db
```

Useful scale-up flags:

```bash
python -m pipeline.run_pipeline \
  --sample-size 25000 \
  --amazon-max-rows 15000 \
  --local-max-rows 10000 \
  --categories Home_and_Kitchen,Electronics,Cell_Phones_and_Accessories \
  --reset-neo4j
```

## Visualization Queries

Sanity dashboard:

```cypher
MATCH (n)
RETURN labels(n)[0] AS label, count(*) AS count
ORDER BY label;
```

Relationship counts:

```cypher
MATCH ()-[r]->()
RETURN type(r) AS relationship, count(*) AS count
ORDER BY relationship;
```

User review neighborhood:

```cypher
MATCH path = (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(i:Item)-[:IN_CATEGORY]->(c:Category)
RETURN path
LIMIT 50;
```

Category distribution:

```cypher
MATCH (i:Item)-[:IN_CATEGORY]->(c:Category)
RETURN c.name AS category, count(i) AS items
ORDER BY items DESC;
```

Nigerian vs Amazon review counts:

```cypher
MATCH (r:Review)
RETURN r.locale AS locale, r.platform AS platform, count(*) AS reviews
ORDER BY reviews DESC;
```

Vector-search smoke test:

```cypher
MATCH (seed:Review)
WHERE seed.embedding IS NOT NULL
WITH seed LIMIT 1
CALL db.index.vector.queryNodes('review_embedding', 5, seed.embedding)
YIELD node, score
RETURN seed.text AS seed_text, node.text AS similar_text, score
ORDER BY score DESC;
```

Start with Neo4j Browser for validation. For presentation visuals, use Neo4j
Bloom if available, or export query results as CSV/GraphML into Gephi.

