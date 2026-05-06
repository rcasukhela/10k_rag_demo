## Retrieval policy notes

Current retrieval uses BM25 over chunk text.

Known limitation: temporal queries such as “in 2022” may retrieve chunks from other filing years if those chunks have stronger lexical overlap.