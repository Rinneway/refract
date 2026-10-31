# Provenance of code and data

## Origin

Refract grew out of the agent component that its author built in an earlier team project on news analytics for business decisions. Other members of that team built the data pipeline, the backend and the user interface.

This repository is a standalone re-implementation of the agent component only.

## What is not here

The earlier project worked with data and documents that cannot be published. This repository contains none of:

- client data — neither raw tables nor profiles built from them, even anonymised; no client identifiers;
- internal research — documents, extracted text, theses, quotes or figures;
- infrastructure — LLM gateway addresses or keys;
- task descriptions and other documents of the earlier project;
- the other team members' code (data pipeline, backend, interface) or texts of news articles collected for that project.

The earlier repository's git history is not carried over.

## What changed compared with the first version

| Area | First version | Refract |
| --- | --- | --- |
| Subjects | corporate clients of one bank | companies, sole proprietors, self-employed people, private individuals |
| Institution | one named bank | any institution, configured by name; a fictional “Demo Bank” in the examples |
| Profile | built from the columns of one dataset | a documented, source-independent `Profile` contract |
| Classification codes, countries | Russian activity codes, Russian country names | NACE Rev. 2 / OKVED 2 divisions, ISO 3166 codes with Russian and English aliases |
| Language | Russian | Russian and English, chosen per run |
| Models | one gateway, a fixed list of models | any OpenAI-compatible endpoint, per-step routing, structured-output fallback |
| Policies | product rules tied to one bank | a generic, editable policy in two languages; products can be switched off |
| Interfaces | the team's backend | Python API, CLI, REST API with SSE, MCP server, RSS collector |
| Evaluation | a script tied to one dataset | a general harness with ablations, baselines, several judges and statistics |
| Number check | context-wide, digits inside identifiers counted | digits glued to identifiers ignored, small percentages checked |

## Synthetic data

Everything in `examples/` was written for this repository: six fictional subjects (two companies, a sole proprietor, a self-employed designer, a household, a UK studio), eight news items marked as coming from a synthetic source, four research notes of the fictional “Demo Bank” and sixteen theses whose quotes are verified against those notes by a test. Numbers and forecasts in them do not describe real events, institutions or people.

## Guard

`tests/test_cli_and_schemas.py::test_no_confidential_material_in_repository` fails if a string shaped like the original data's client identifiers or like an API key appears anywhere in the repository, and checks that `.env` is ignored.
