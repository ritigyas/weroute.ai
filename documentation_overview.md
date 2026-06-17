# Executive Summary

**Project:** `weroute.ai`

**Purpose & Business Objective**
- Provide a lightweight, automated solution for network operations teams to collect, store, and visualise health metrics of Juniper (and potentially other) network devices.
- Reduce manual effort and latency in assessing device reachability, configuration state (NETCONF), BGP health, interface status, and environmental parameters.
- Enable on‑demand ICMP ping checks that update reachability information without triggering a full NETCONF collection.

**Target Users**
- Network Operations Engineers (NOC/SRE) who need a quick health‑dashboard.
- Platform engineers extending the collector for additional device families.
- Incident responders who need a snapshot of the fleet at a given point in time.

**High‑Level Value**
- Centralised, always‑up‑to‑date view of device health.
- Fast, asynchronous collection via Ansible Playbooks.
- Historical run‑ID based snapshots that can be purged automatically.
- Extensible architecture (FastAPI backend, Streamlit UI, pluggable Ansible playbooks).
