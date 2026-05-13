# Incident Runbook (Internal)

## Severity Levels
- SEV0: Widespread outage or data integrity risk. Immediate paging required.
- SEV1: Major degradation impacting many users.
- SEV2: Partial degradation or localized outage.
- SEV3: Minor issue / workaround available.

## First 5 Minutes (Checklist)
1. Confirm impact scope: who/what/where is affected?
2. Assign roles: Incident Commander (IC), Communications Lead, Scribe.
3. Create incident channel and incident ticket.
4. Start timeline notes (time-stamped).
5. Stabilize: mitigate blast radius, rollback if needed.

## Triage Steps
- Check recent deploys and feature flags.
- Check key health dashboards: latency, errors, saturation.
- Identify the failing dependency (db, cache, queue, external API).
- Apply mitigations: scale, rollback, feature flag off, rate limiting.

## Communication
- Internal updates every 15 minutes for SEV0/SEV1.
- External status page updates every 30 minutes for SEV0/SEV1.

## Post-Incident
- Draft postmortem within 48 hours.
- Include: root cause, contributing factors, detection gaps, corrective actions, owners, timelines.