# tranz

Point-embedding knowledge graph completion: TransE, RotatE, ComplEx.

Entities are points in vector space. Relations are transformations (translation, rotation, diagonal scaling). Training via negative sampling with log-sigmoid loss and self-adversarial weighting.

Uses `subsume` for dataset loading and evaluation infrastructure. GPU training via candle.

Dual-licensed under MIT or Apache-2.0.

## Models

| Model | Relation transform | Space | Reference |
|---|---|---|---|
| TransE | Translation | Real | Bordes et al., 2013 |
| RotatE | Rotation | Complex | Sun et al., 2019 |
| ComplEx | Diagonal | Complex | Trouillon et al., 2016 |
| DistMult | Diagonal | Real | Yang et al., 2015 |

## Relationship to subsume

`subsume` embeds entities as geometric regions (boxes, cones) where containment encodes subsumption. `tranz` embeds entities as points where distance/similarity encodes relational facts. Different geometric paradigms for different tasks:

- **subsume**: ontology completion, taxonomy expansion, logical query answering
- **tranz**: link prediction, relation extraction, knowledge base completion
