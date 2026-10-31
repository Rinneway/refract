# Synthetic demo data

**Everything in this folder is invented** for demonstrations and tests of Refract. The subjects, news items, research notes and the “Demo Bank” do not exist; numbers and forecasts do not describe real events, institutions or people. News items name a synthetic source and link to `example.org`.

| Folder | Content |
| --- | --- |
| `profiles/` | six subjects of all four kinds |
| `events/` | eight news items, five in Russian and three in English, already tagged |
| `knowledge/docs/` | four research notes of Demo Bank (two in Russian, two in English) |
| `knowledge/theses.json` | sixteen theses; every quote is verified against its note by `tests/test_knowledge.py` |
| `pairs.jsonl` | all 96 subject × event × language pairs for `refract eval` |
| `pairs_parity.jsonl` | 7 subject × event × role triples in both languages, for checking the English prompts against the Russian ones |

## Subjects

| Id | Kind | Who |
| --- | --- | --- |
| `demo_building_materials` | company | building-materials wholesaler, Yekaterinburg; loans, leasing, imports from China; its largest customer is a developer |
| `demo_grain_farm` | company | grain farm, Krasnodar region; exports to Turkey and Egypt; a grain trader buys 38% |
| `demo_london_studio` | company | software studio, London; 45% of incoming payments from a US client in dollars |
| `demo_coffee_ip` | sole proprietor | coffee shop in Kazan on the simplified tax regime; card payments, a small equipment loan |
| `demo_designer_npd` | self-employed | graphic designer in St Petersburg on the professional-income tax; clients in Kazakhstan and the UAE |
| `demo_family_saver` | individual | household in Novosibirsk with a floating-rate mortgage and a deposit, planning a bakery |

## News

| Id | Lang | Topic | Interesting for |
| --- | --- | --- | --- |
| `ev_rate_hold_ru` | ru | the central bank holds the rate at 14% | borrowers, savers |
| `ev_yuan_ru` | ru | the yuan falls against the rouble | importers from China |
| `ev_usn_vat_ru` | ru | a lower VAT threshold for the simplified regime | sole proprietors in food service and retail |
| `ev_sme_loans_ru` | ru | subsidised loans extended to cafés and bakeries | the coffee shop, the household's bakery plan |
| `ev_injection_ru` | ru | a coffee chain closes outlets — **contains a prompt injection** | safety tests |
| `ev_wheat_en` | en | wheat export prices rise as freight slows | the grain farm |
| `ev_housing_en` | en | new-build sales fall | the wholesaler via its developer customer |
| `ev_dollar_en` | en | the dollar strengthens on US inflation | the London studio, the designer |
