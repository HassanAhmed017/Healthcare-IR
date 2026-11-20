# IR-Healthcare

Information Storage and Retrieval project (Healthcare version).
Positional index built with Spark, then TF, IDF, TF-IDF, normalization,
phrase queries, Boolean queries, similarity and ranking.

## Requirements
- Python 3.10+
- Java 8 or 11 or 17 (required by Spark)

## Setup
pip install -r requirements.txt

## Run
python src/main.py

## Structure
- data/         the 10 documents (d1.txt to d10.txt)
- src/          core logic (index, retrieval, queries, main)
- gui/          optional interface
- output/       generated results
- screenshots/  program output images
- docs/         project requirements and diagrams