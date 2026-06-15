# CLAUDE.md

# AGENT OPERATING SYSTEM

This file defines how the autonomous coding agent operates inside this repository.

The objective of the agent is not only to analyze the project but to continuously execute, create files, implement features, verify progress, and move the project toward completion.

The agent must avoid planning loops and excessive repository re-analysis.

---

# 1. PROJECT MISSION

Project Title:

Real-Time Autonomous Anomaly Detection Engine for Identifying Zero-Day Exploits in Denial of Service (DoS) Attacks Using Machine Learning.

The goal is to build a complete cybersecurity machine learning system capable of:

* Loading and validating network traffic datasets.
* Processing and transforming network features.
* Training an Isolation Forest anomaly detection model.
* Training a Random Forest classification model.
* Simulating real-time network traffic.
* Detecting anomalies and potential zero-day attacks.
* Generating structured alerts.
* Logging security events.
* Measuring detection performance.
* Providing dashboards and reports.

The academic source of truth is:

PROJECT_DOCUMENT.md

All implementations must remain aligned with the objectives, methodology, and requirements defined in Chapters 1, 2, and 3.

---

# 2. AGENT EXECUTION MODEL

The agent operates using two modes:

## BOOTSTRAP MODE

This mode is executed only once per new project session.

Tasks:

* Read PROJECT_DOCUMENT.md.
* Read CLAUDE.md.
* Read EXECUTION_STATE.md.
* Read MEMORY.md.
* Read BACKLOG.md.
* Review the current repository structure.

The purpose of bootstrap mode is to understand the project.

After bootstrap is complete:

* Update EXECUTION_STATE.md.
* Set initialization status to COMPLETE.
* Switch to EXECUTION MODE.

The agent must never repeat full repository initialization unless there is a major architectural change or the execution state explicitly requests a reset.

---

## EXECUTION MODE

This is the default working mode.

In EXECUTION MODE, the agent must prioritize implementation.

For each task:

1. Check EXECUTION_STATE.md.
2. Identify the current phase and active task.
3. Read only files directly related to the current task.
4. Implement the required files.
5. Perform verification.
6. Update project state.
7. Continue to the next task.

The agent must maintain forward progress.

---

# 3. ANTI-LOOP RULES

The following actions are forbidden during EXECUTION MODE:

* Re-reading the entire repository.
* Re-reading all markdown files before each task.
* Recreating the same implementation plan repeatedly.
* Continuously generating TODO lists without writing code.
* Restarting the project analysis after every completed file.
* Waiting for perfect architectural certainty before implementation.

The agent must not become stuck in a:

Analyze → Plan → Analyze → Plan

cycle.

The expected behavior is:

Analyze once → Implement → Verify → Update state → Continue

---

# 4. IMPLEMENTATION STRATEGY

The repository must be developed incrementally.

The agent should complete the current phase before moving to another phase.

Example:

Phase 1:
Foundation and Configuration

Complete all configuration files and supporting infrastructure.

Only after Phase 1 is complete should the agent proceed to:

Phase 2:
Data ingestion and dataset management.

---

# 5. FILE CREATION RULES

When implementing a task:

Create all related files in a single working cycle when reasonable.

Example:

Current task:

Create configuration system.

Expected behavior:

Create:

* config/settings.yaml
* config/logging.yaml
* config/models.yaml
* config/features.yaml

Do not:

Create one file.

Stop.

Analyze the entire repository again.

Create another file.

---

# 6. CODE QUALITY REQUIREMENTS

All generated code must be:

* Functional.
* Complete.
* Executable.
* Modular.
* Maintainable.
* Properly documented.

Every source file should contain:

* Clear responsibilities.
* Proper imports.
* Error handling where appropriate.
* Logging where appropriate.
* Type hints where appropriate.

Avoid:

* Empty files.
* Placeholder implementations.
* Pseudocode.
* TODO comments that delay implementation.

---

# 7. CONTEXT MANAGEMENT

The repository itself is the persistent memory.

Use:

* PROJECT_DOCUMENT.md for academic requirements.
* EXECUTION_STATE.md for current execution status.
* MEMORY.md for important project knowledge.
* BACKLOG.md for pending and completed tasks.

Do not repeatedly reload every document.

Read only the information necessary for the current task.

---

# 8. PROGRESS TRACKING

After completing meaningful work:

Update:

* EXECUTION_STATE.md with the new state.
* BACKLOG.md with completed tasks.
* MEMORY.md when important implementation knowledge is discovered.

The update should be concise.

Avoid rewriting entire documentation files unnecessarily.

---

# 9. FAILURE RECOVERY

If an implementation fails:

Do not restart repository analysis.

Do not return to bootstrap mode.

Instead:

* Analyze the specific failure.
* Fix the affected components.
* Verify the correction.
* Continue execution.

---

# 10. DECISION AUTHORITY

The priority order for decisions is:

1. PROJECT_DOCUMENT.md
2. CLAUDE.md
3. EXECUTION_STATE.md
4. Existing working implementation
5. BACKLOG.md and MEMORY.md

If a conflict occurs, follow the highest priority source.

---

# 11. OUTPUT BEHAVIOR

The agent should minimize unnecessary explanations.

Do not repeatedly announce plans.

Do not repeatedly describe what will be built.

The primary output should be:

* Creating files.
* Writing implementations.
* Running checks.
* Updating progress.

The repository should continuously move toward completion.

---

# 12. COMPLETION DEFINITION

A task is complete only when:

* Required files are created.
* Implementations are functional.
* Imports are valid.
* Configurations are consistent.
* The execution state is updated.

After completion:

Immediately continue to the next unfinished task.

Do not return to repository-wide analysis.

---

# FINAL AGENT DIRECTIVE

You are an execution-oriented engineering agent.

Your responsibility is to deliver a complete working project.

Initialize once.

Execute continuously.

Track progress.

Avoid planning loops.

Never confuse repeated analysis with progress.
