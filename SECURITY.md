# Security

Please report vulnerabilities privately through GitHub's “Report a vulnerability” (Security → Advisories) rather than in a public issue.

Areas of particular interest:

- prompt injection through news text or user-provided profile fields that changes the impact level, the products offered or the statements attributed to the institution;
- a tool call that reads data of a subject other than the one being analysed;
- leaking keys or private data through logs, traces or API responses.

Keys belong in environment variables; the REST API can require a bearer token (`REFRACT_API_TOKEN`) and should not be exposed to the internet without one.
