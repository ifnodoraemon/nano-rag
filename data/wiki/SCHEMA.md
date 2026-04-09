# Nano RAG Wiki Schema

This directory is the compiled knowledge layer between raw source files and query-time retrieval.

Structure:
- `sources/`: one markdown page per ingested source document
- `topics/`: synthesized topic pages aggregated from compiled sources
- `index.md`: catalog of compiled pages
- `log.md`: append-only ingest timeline

Conventions:
- Raw sources remain immutable in `data/raw/`
- Wiki pages are regenerated or updated by the ingestion pipeline
- `index.md` should be readable first when exploring the wiki layer
