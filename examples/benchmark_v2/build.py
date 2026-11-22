"""Authoring source for the fictional v2 fixtures; never used by retrieval at runtime."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS = {}
PASSAGES = {}
QUESTIONS = []


def document(split, name, title, paragraphs):
    name = f"{split}_{name}.md"
    text = "# " + title + "\n\n"
    for key, paragraph in paragraphs:
        start = len(text)
        text += paragraph + "\n\n"
        PASSAGES[f"{split}.{key}"] = dict(
            document=name,
            split=split,
            quote=paragraph,
            char_start=start,
            char_end=start + len(paragraph),
        )
    DOCS[name] = text


def question(
    split, category, prompt, answer, keys=(), alternatives=(), behavior="answer", calculation=None
):
    # Each group is required; IDs within one group are alternative supporting passages.
    groups = [[f"{split}.{key}"] for key in keys]
    groups += [[f"{split}.{key}" for key in group] for group in alternatives]
    item = dict(
        id=f"v2_{len(QUESTIONS) + 1:03}",
        split=split,
        category=category,
        question=prompt,
        expected_answer=answer,
        expected_behavior=behavior,
        is_unanswerable=behavior == "abstain",
        evidence_groups=groups,
    )
    if calculation:
        item["calculation"] = calculation
    QUESTIONS.append(item)


document(
    "dev",
    "routing",
    "Aster Robotics: service ownership and routing register",
    [
        (
            "route_unit",
            "The unit ORBIT-17 is an inspection rover operated by Aster Robotics. Its support contract code is SC-COPPER, not SC-SILVER.",
        ),
        (
            "route_owner",
            "ORBIT-17 belongs to the Field Autonomy group. The accountable incident owner for Field Autonomy is the duty coordinator.",
        ),
        (
            "route_page",
            "To reach the Field Autonomy duty coordinator, use paging route field-auto-primary. The register intentionally omits personal phone numbers.",
        ),
        (
            "route_rule",
            "Contract codes identify service entitlements; asset nicknames and purchase prices do not determine the support response window.",
        ),
        (
            "route_diag",
            "Diagnostics for ORBIT-17 must be captured before a motor controller is reset. Capture the controller event log and the battery telemetry bundle.",
        ),
        (
            "route_secondary",
            "The rover ORBIT-71 is a different asset assigned to Warehouse Automation. It uses contract SC-SILVER and paging route warehouse-ops.",
        ),
        (
            "route_scope",
            "This register covers operational ownership only. Scheduling, replacement parts and commercial discounts require their respective control documents.",
        ),
    ],
)
document(
    "dev",
    "contracts",
    "Aster Robotics: active service contract matrix",
    [
        (
            "contract_copper",
            "SC-COPPER provides a critical-incident acknowledgement target of 25 minutes and a scheduled preventive visit every 6 weeks.",
        ),
        (
            "contract_silver",
            "SC-SILVER provides a critical-incident acknowledgement target of 90 minutes and a preventive visit every 12 weeks.",
        ),
        (
            "contract_hours",
            "SC-COPPER support operates around the clock. SC-SILVER support operates Monday through Friday from 08:00 to 18:00 UTC, excluding holidays.",
        ),
        (
            "contract_restoration",
            "Neither SC-COPPER nor SC-SILVER promises a restoration deadline. An acknowledgement records receipt; it does not mean the rover is repaired.",
        ),
        (
            "contract_charge",
            "Contract coverage excludes consumable batteries. Replacement batteries require a separately approved parts order.",
        ),
        (
            "contract_expiry",
            "Active contract entitlements override catalog descriptions for the specific asset. Expired contract proposals are not authority for current entitlements.",
        ),
        (
            "contract_scope",
            "This matrix specifies support response and visit schedules. It contains no payroll, customer profitability or individual contact records.",
        ),
    ],
)
document(
    "dev",
    "parts",
    "Aster Robotics: approved parts catalog",
    [
        (
            "part_code",
            "The approved replacement battery for ORBIT-17 is pack BP-42. Pack BP-24 fits ORBIT-71 and must not be fitted to ORBIT-17.",
        ),
        (
            "part_price",
            "BP-42 has a list price of EUR 480 per pack. Each pack contains 8 cells and is shipped with a reusable protective case.",
        ),
        (
            "part_discount",
            "Orders of at least 10 BP-42 packs receive a 15% discount on the pack price. Freight is EUR 35 per order and is not discounted.",
        ),
        (
            "part_stock",
            "The catalog stock ledger lists 36 BP-42 packs available on 4 February 2026. This is a dated stock count, not a promise of future availability.",
        ),
        (
            "part_safety",
            "Battery packs may be installed only after the rover has been isolated from charging equipment. The protective case is not an electrical isolation device.",
        ),
        (
            "part_warranty",
            "Unopened BP-42 packs have a 12-month warranty from dispatch. Installing a pack does not extend the original dispatch-based warranty.",
        ),
        (
            "part_scope",
            "Prices exclude tax. No applicable tax rate is specified, so a tax-inclusive total cannot be derived from this catalog.",
        ),
    ],
)
document(
    "dev",
    "shipping",
    "Aster Robotics: transport and dispatch instructions",
    [
        (
            "ship_class",
            "BP-42 battery packs use transport class DG-9. DG-9 shipments require a signed dangerous-goods declaration.",
        ),
        (
            "ship_route",
            "DG-9 shipments leave the East depot on Tuesdays and Thursdays. Ordinary replacement wheels may leave on any working day.",
        ),
        (
            "ship_lead",
            "After approval, domestic DG-9 orders require 3 working days for preparation before the next eligible dispatch day.",
        ),
        (
            "ship_label",
            "Each DG-9 shipping case must show the pack model, dispatch date and handling orientation. Customer account passwords must never appear on the label.",
        ),
        (
            "ship_contact",
            "The dispatch queue owner is Logistics Control. Finance approval and signed transport paperwork are independent gates before release.",
        ),
        (
            "ship_proof",
            "Proof of dispatch is the carrier acceptance scan. Printing a label alone does not establish that a pack has left the depot.",
        ),
        (
            "ship_scope",
            "International customs duties and country-specific transport exceptions are outside this domestic instruction.",
        ),
    ],
)
document(
    "dev",
    "safety_2024",
    "Aster Robotics: archived workshop safety rules, 2024 edition",
    [
        (
            "old_status",
            "The 2024 workshop safety edition expired on 31 December 2025. It is retained for historical audits and does not govern work in 2026.",
        ),
        (
            "old_check",
            "Under the 2024 edition, a rover returning from outdoor service required an insulation check every 120 operating hours.",
        ),
        (
            "old_sign",
            "The 2024 edition allowed the shift supervisor alone to authorize a return to service.",
        ),
        ("old_temp", "The 2024 charging-room temperature ceiling was 32 degrees Celsius."),
        (
            "old_archive",
            "Historical inspection sheets record the edition used at the time. Do not silently replace a historical threshold with a newer threshold in audit records.",
        ),
        (
            "old_examples",
            "An audit of a November 2025 repair must use the rules effective in November 2025, even if the audit itself happens the following year.",
        ),
    ],
)
document(
    "dev",
    "safety_2026",
    "Aster Robotics: current workshop safety rules, effective 1 January 2026",
    [
        (
            "new_check",
            "From 1 January 2026, a rover returning from outdoor service requires an insulation check every 80 operating hours.",
        ),
        (
            "new_sign",
            "Return to service requires two approvals: the shift supervisor and a certified electrical inspector. Neither approval alone is sufficient.",
        ),
        (
            "new_temp",
            "Charging must pause when the room temperature exceeds 28 degrees Celsius. Work may resume only after the room has cooled to 26 degrees Celsius or below.",
        ),
        (
            "new_fail",
            "A failed insulation check requires quarantine in Bay Q2 and a new inspection after repair. Resetting the inspection timer does not clear a failed result.",
        ),
        (
            "new_scope",
            "These rules govern all Aster workshop shifts from their effective date. They replace the 2024 edition, including its single-approver exception.",
        ),
        (
            "new_records",
            "The inspection ledger must record the asset ID, operating-hour counter, result and both approvers. Photographs supplement but do not replace those fields.",
        ),
    ],
)
document(
    "dev",
    "finance",
    "Aster Robotics: management accounts for fiscal 2025",
    [
        (
            "fin_revenue",
            "Fiscal 2025 revenue was EUR 3.6 million, compared with EUR 3.0 million in fiscal 2024. These are recognized revenues, not order bookings.",
        ),
        (
            "fin_profit",
            "Fiscal 2025 operating costs were EUR 2.7 million and operating profit was EUR 0.9 million. Finance charges are excluded from operating profit.",
        ),
        (
            "fin_segments",
            "Service contracts produced EUR 2.16 million of fiscal 2025 revenue. Hardware sales produced the remaining EUR 1.44 million.",
        ),
        (
            "fin_orders",
            "The fiscal 2025 year-end order backlog was EUR 1.1 million. Backlog is not included in recognized revenue until delivery criteria are met.",
        ),
        (
            "fin_currency",
            "All monetary figures in these accounts are in euros. No foreign exchange rates or geographic revenue breakdowns are included.",
        ),
        (
            "fin_audit",
            "These fictional management accounts are unaudited. Figures are provided as a calculation exercise and should not be interpreted as an investment recommendation.",
        ),
    ],
)
document(
    "dev",
    "records",
    "Aster Robotics: information retention schedule",
    [
        (
            "ret_telemetry",
            "Routine rover telemetry is retained for 45 days after collection. Incident-linked telemetry is retained for 18 months after incident closure.",
        ),
        (
            "ret_invoices",
            "Supplier invoices are retained for 7 years after the end of the relevant fiscal year. Unapproved draft purchase requests are retained for 60 days.",
        ),
        (
            "ret_access",
            "Temporary export links expire after 48 hours. Expiry of a link does not delete the retained underlying record.",
        ),
        (
            "ret_owner",
            "Records Operations owns the retention schedule. A legal hold suspends deletion until Legal explicitly releases the hold.",
        ),
        (
            "ret_priority",
            "Where an incident tag and a routine telemetry tag both apply, the incident-linked retention period takes precedence.",
        ),
        (
            "ret_scope",
            "The schedule does not contain biometric retention rules, encryption secrets or employee benefit records. Those subjects must not be inferred from telemetry durations.",
        ),
    ],
)
document(
    "dev",
    "operations",
    "Aster Robotics: canary release procedure",
    [
        (
            "ops_flag",
            "Feature switch ROVER_SAFE_PATH defaults to enabled in production. Its documented values are enabled and disabled, not numeric levels.",
        ),
        (
            "ops_probe",
            "The rover gateway readiness probe is GET /ready/rover. GET /health/rover reports process liveness and must not be substituted for readiness.",
        ),
        (
            "ops_canary",
            "A release begins with 4 rovers in the canary cohort and an observation window of 35 minutes.",
        ),
        (
            "ops_abort",
            "Stop rollout when command rejection exceeds 1.5% for 4 consecutive minutes. A single transient sample does not satisfy this stopping condition.",
        ),
        (
            "ops_rollback",
            "Rollback restores the preceding firmware image. It does not undo calibration changes stored on the motor controller.",
        ),
        (
            "ops_review",
            "After rollback, retain the failed-image identifier and observation logs. The release engineer records the result before scheduling a new canary.",
        ),
    ],
)
document(
    "dev",
    "training",
    "Aster Robotics: competence and workshop access register",
    [
        (
            "train_pack",
            "BP-42 handling requires certification BAT-L2. General mechanical induction alone does not authorize battery-pack handling.",
        ),
        (
            "train_valid",
            "BAT-L2 certification remains valid for 9 months from assessment. A failed reassessment immediately suspends pack-handling access.",
        ),
        (
            "train_exam",
            "The BAT-L2 assessment contains 24 questions; a candidate must answer at least 21 correctly to pass.",
        ),
        (
            "train_booking",
            "Assessment bookings close 5 working days before the session. Supervisors may request a later session but cannot waive the assessment.",
        ),
        (
            "train_register",
            "The competence register records certificate code and expiry date, not personal exam answers. Access checks must use the certificate expiry, not the booking date.",
        ),
        (
            "train_scope",
            "The training schedule is separate from product warranty and support entitlements. A long support contract does not extend a technician certificate.",
        ),
    ],
)
document(
    "dev",
    "proposal",
    "Aster Robotics: rejected commercial proposal, November 2025",
    [
        (
            "distract_proposal",
            "This rejected draft proposed SC-COPPER acknowledgement in 10 minutes and BP-42 packs at EUR 400. Neither proposal was approved or made effective.",
        ),
        (
            "distract_proposal2",
            "The draft requested a 25% bulk-order discount. Procurement declined that request and instructed staff to use the approved parts catalog.",
        ),
        (
            "distract_proposal3",
            "This document is a deliberately obsolete distractor. An exact product-name match does not make its commercial terms authoritative.",
        ),
    ],
)
document(
    "dev",
    "warehouse",
    "Aster Robotics: warehouse-only equipment notice",
    [
        (
            "distract_warehouse",
            "Warehouse trolley ORBIT-170 is not rover ORBIT-17. The trolley uses cell cartridge BP-420 and does not accept rover pack BP-42.",
        ),
        (
            "distract_warehouse2",
            "BP-420 cartridges are priced at EUR 75 and have no relationship to the BP-42 rover-pack price or discount.",
        ),
        (
            "distract_warehouse3",
            "Warehouse staff retain trolley route logs for 10 days. That period is not a retention rule for rover telemetry.",
        ),
    ],
)

document(
    "held_out",
    "asset_map",
    "Bracken Observatory: instrument and programme map",
    [
        (
            "map_instrument",
            "The instrument LUMEN-8 is assigned to programme NIGHTGLASS. Its detector assembly is model CCD-K7.",
        ),
        (
            "map_programme",
            "Programme NIGHTGLASS is funded by grant GR-204. The programme lead is the observing coordinator, not the facilities manager.",
        ),
        (
            "map_room",
            "LUMEN-8 is installed in dome D3. The visually similar guide camera LUMEN-80 is installed in dome D1 and is not part of NIGHTGLASS.",
        ),
        (
            "map_schedule",
            "NIGHTGLASS observing blocks start at 22:30 UTC. Clock times in this register use UTC throughout the year without daylight-saving adjustments.",
        ),
        (
            "map_scope",
            "This register links instruments, programmes and locations. It does not specify grant purchase limits or detector procurement procedures.",
        ),
        (
            "map_tracking",
            "Staff must record the exact instrument suffix in fault reports. Dropping the zero from LUMEN-80 can send a guide-camera issue to the wrong instrument team.",
        ),
    ],
)
document(
    "held_out",
    "grants",
    "Bracken Observatory: current research grant conditions",
    [
        (
            "grant_limit",
            "Grant GR-204 requires two written quotations for any single equipment purchase above GBP 2500. Splitting one purchase into invoices does not remove this requirement.",
        ),
        (
            "grant_quota",
            "GR-204 allocates 160 observing hours per quarter. Unused hours expire at quarter end and cannot be carried forward.",
        ),
        (
            "grant_window",
            "GR-204 eligible expenditure must be incurred between 1 April 2026 and 31 March 2027. Purchase authorization alone is not evidence that expenditure was incurred.",
        ),
        (
            "grant_sign",
            "The research finance officer must approve GR-204 equipment purchases. Approval by the observing coordinator alone is insufficient.",
        ),
        (
            "grant_scope",
            "Grant GR-811 belongs to the solar-monitoring programme and has different conditions. GR-204 limits must not be copied into a GR-811 approval record.",
        ),
        (
            "grant_mirror",
            "For GR-204, the current instrument allocation is 160 observing hours in each quarter; unused allocation expires at that quarter end.",
        ),
    ],
)
document(
    "held_out",
    "detectors",
    "Bracken Observatory: detector procurement sheet",
    [
        (
            "detector_vendor",
            "The sole approved supplier of CCD-K7 detector assemblies is Meridian Optics. The approved procurement route is a sealed purchase order to that supplier.",
        ),
        (
            "detector_price",
            "One CCD-K7 assembly costs GBP 3200 before tax and freight. The procurement sheet does not state a tax rate or a freight charge.",
        ),
        (
            "detector_lead",
            "CCD-K7 quoted manufacturing lead time is 7 weeks from supplier order acceptance. Delivery transit time is additional and is not specified.",
        ),
        (
            "detector_maintenance",
            "A CCD-K7 assembly requires a calibration exposure sequence every 30 nights of use. Calendar months are not a substitute for counting nights of use.",
        ),
        (
            "detector_warranty",
            "The CCD-K7 warranty is 18 months from acceptance testing. Accidental damage is excluded; routine calibration remains the observatory responsibility.",
        ),
        (
            "detector_alternate",
            "CCD-K70 assemblies are an incompatible product and are not approved replacements for CCD-K7. Similar naming is not a substitution authorization.",
        ),
    ],
)
document(
    "held_out",
    "vendor_register",
    "Bracken Observatory: supplier payment and dispatch register",
    [
        (
            "vendor_deposit",
            "Meridian Optics requires a 30% deposit when accepting a CCD-K7 order. The balance is payable after acceptance testing.",
        ),
        (
            "vendor_dispatch",
            "Meridian Optics dispatches detector assemblies from its North quay laboratory. The South quay site handles mirrors only.",
        ),
        (
            "vendor_payment",
            "Supplier bank details are withheld from this public testing corpus. Staff must obtain verified details through the finance system.",
        ),
        (
            "vendor_reference",
            "Purchase orders must include the instrument model and grant code. A delivery address alone is not enough to identify the funding authorization.",
        ),
        (
            "vendor_scope",
            "A supplier deposit is part of the assembly price, not a surcharge. The register does not override grant quotation thresholds or approval roles.",
        ),
        (
            "vendor_change",
            "A request to change supplier payment details must be independently verified by the research finance officer before any payment is released.",
        ),
    ],
)
document(
    "held_out",
    "access_2023",
    "Bracken Observatory: retired visitor access rules, 2023 edition",
    [
        (
            "access_old",
            "The 2023 visitor rules permitted unescorted dome access after a 20-minute induction. This edition ceased to apply on 30 June 2026.",
        ),
        (
            "access_old_hours",
            "Under the 2023 rules, visitor badges remained valid for 72 hours from issue. Audit records for historical visits must retain the original issue time.",
        ),
        (
            "access_old_weather",
            "The 2023 edition closed external observing platforms at sustained wind speeds above 45 km/h.",
        ),
        (
            "access_old_authority",
            "This retired edition is stored for incident reconstruction. It must not be used to authorize a new visit after its expiry date.",
        ),
        (
            "access_old_book",
            "The historical visitor book records the host and visit date. It contains no current emergency permissions.",
        ),
    ],
)
document(
    "held_out",
    "access_2026",
    "Bracken Observatory: visitor rules effective 1 July 2026",
    [
        (
            "access_new",
            "From 1 July 2026, all visitors in domes must be accompanied by an accredited host. Completing an induction does not waive the escort requirement.",
        ),
        (
            "access_new_hours",
            "Visitor badges issued under the July 2026 rules expire 12 hours after issue. A host cannot extend a badge by writing a new expiry on it.",
        ),
        (
            "access_new_weather",
            "External observing platforms close when sustained wind exceeds 35 km/h. A weather closure takes precedence over a previously approved visitor booking.",
        ),
        (
            "access_new_after",
            "After a weather closure, the facilities controller must authorize reopening. A visitor host is not authorized to reopen the platform.",
        ),
        (
            "access_new_scope",
            "These rules supersede the 2023 edition for visits from 1 July 2026. They do not change the historical record of visits completed before that date.",
        ),
        (
            "access_new_record",
            "Hosts record visitor entry and exit in the live register. Badge issue time and visitor arrival time are separate fields and must not be conflated.",
        ),
    ],
)
document(
    "held_out",
    "archive",
    "Bracken Observatory: scientific data archive specification",
    [
        (
            "arc_raw",
            "Raw spectrograph frames are retained for 120 days after acquisition. Calibrated science products are retained for 6 years after publication.",
        ),
        (
            "arc_embargo",
            "NIGHTGLASS calibrated products carry an embargo of 9 months after acquisition. Embargo expiry controls public access, not the retention clock.",
        ),
        (
            "arc_code",
            "Archive collection identifier NG-CAL-6Y selects calibrated NIGHTGLASS products. NG-RAW-120D selects the raw-frame collection.",
        ),
        (
            "arc_checksum",
            "The archive verifies SHA-256 checksums when accepting an upload. A matching checksum detects byte changes but does not prove scientific correctness.",
        ),
        (
            "arc_process",
            "To retrieve an embargoed product, first authenticate the researcher, then verify programme membership, then issue a time-limited download link.",
        ),
        (
            "arc_membership",
            "Embargoed NIGHTGLASS products may be accessed by members of programme NIGHTGLASS. Funding-agency staff do not automatically inherit programme membership.",
        ),
        (
            "arc_scope",
            "The archive specification describes retention and access only. It contains no scientific discovery dates or undisclosed observational results.",
        ),
    ],
)
document(
    "held_out",
    "maintenance",
    "Bracken Observatory: dome equipment maintenance log and procedures",
    [
        (
            "maint_fault",
            "Fault FLT-208 concerns the coolant pump for LUMEN-8. It requires a coolant isolation check before any pump restart.",
        ),
        (
            "maint_fluid",
            "The LUMEN-8 coolant loop holds 24 litres when full. After draining 6 litres, refill the loop to its full level before the isolation test.",
        ),
        (
            "maint_lock",
            "A failed coolant isolation check requires lockout under tag COOL-LOCK. Only a certified cooling technician may release COOL-LOCK after a passing retest.",
        ),
        (
            "maint_interval",
            "Dome D3 shutter seals are inspected every 45 days. The detector calibration interval is controlled by the detector procurement sheet, not by this shutter schedule.",
        ),
        (
            "maint_queue",
            "Routine maintenance requests enter queue OBS-MAINT. A coolant leak during observing is escalated immediately to the duty facilities controller.",
        ),
        (
            "maint_scope",
            "The maintenance log does not list pump serial numbers or spare-part shelf counts. Technicians must consult the asset store for those details.",
        ),
    ],
)
document(
    "held_out",
    "capacity",
    "Bracken Observatory: observing capacity and budget statement",
    [
        (
            "cap_hours",
            "During quarter Q2, usable observing capacity was 240 hours and executed observations used 180 hours. Weather losses are already excluded from usable capacity.",
        ),
        (
            "cap_spend",
            "The Q2 instrument budget was GBP 48000 and recorded instrument expenditure was GBP 36000. Unspent budget is not an authorization to bypass grant controls.",
        ),
        (
            "cap_energy",
            "Dome D3 consumed 1500 kWh in April and 1800 kWh in May. These figures cover the dome supply, not only the LUMEN-8 detector.",
        ),
        (
            "cap_nights",
            "Q2 included 30 scheduled observing nights and 6 were cancelled. A partially completed night still counts as a scheduled night in this statement.",
        ),
        (
            "cap_scope",
            "This fictional statement excludes electricity unit prices, staffing costs and visitor revenue. Those quantities cannot be recovered from energy or hour totals.",
        ),
        (
            "cap_method",
            "Executed hours count time actually spent observing. Requested hours and scheduled hours are separate planning measures and are not reported here.",
        ),
    ],
)
document(
    "held_out",
    "network",
    "Bracken Observatory: telemetry service configuration notes",
    [
        (
            "net_port",
            "The telescope telemetry collector listens on TCP port 7443. Port 7444 is reserved for a maintenance console, not the collector.",
        ),
        (
            "net_token",
            "Setting OBS_LINK_TTL_SECONDS has value 900. It controls archive download-link lifetime in seconds and does not set scientific data retention.",
        ),
        (
            "net_retry",
            "Telemetry clients wait 12 seconds before the first retry, then 24 seconds before the second retry. No additional retries are permitted by this profile.",
        ),
        (
            "net_probe",
            "The telemetry readiness endpoint is /observe/ready. A successful TCP connection alone is not a readiness assertion.",
        ),
        (
            "net_window",
            "The collector accepts a maximum of 80 frames in one upload batch. Larger uploads must be divided without changing individual frame checksums.",
        ),
        (
            "net_scope",
            "This configuration note includes public setting names only. Authentication tokens, host certificates and signing keys are deliberately absent.",
        ),
    ],
)
document(
    "held_out",
    "draft",
    "Bracken Observatory: unapproved NIGHTGLASS expansion bid",
    [
        (
            "held_draft",
            "This unapproved bid requested 300 quarterly hours for NIGHTGLASS and detector assemblies at GBP 2800. Neither value is an approved grant or procurement term.",
        ),
        (
            "held_draft2",
            "The bid proposed waiving written quotations and allowing unescorted visitors after dark. These proposals did not become operational rules.",
        ),
        (
            "held_draft3",
            "The bid is retained as a search distractor; approved registers and dated effective rules take precedence.",
        ),
    ],
)
document(
    "held_out",
    "guidecamera",
    "Bracken Observatory: guide-camera quick sheet",
    [
        (
            "held_guide",
            "LUMEN-80 is a guide camera in dome D1. Its detector model is CMOS-J2, not CCD-K7, and its programme is DAYFINDER.",
        ),
        (
            "held_guide2",
            "DAYFINDER uses grant GR-811. Its allocated hours are not the NIGHTGLASS allocation, and this sheet does not disclose GR-811 purchase limits.",
        ),
        (
            "held_guide3",
            "The guide-camera store retains preview thumbnails for 14 days. Thumbnails are not the raw spectrograph frames governed by the scientific archive.",
        ),
    ],
)


# Development: robotics operations, purchasing and workshop controls.
def q(cat, prompt, answer, keys=(), **kw):
    question("dev", cat, prompt, answer, keys, **kw)


q(
    "factual",
    "Which team owns ORBIT-17, and which role is accountable for its incidents?",
    "Field Autonomy owns it; the duty coordinator is accountable.",
    ["route_owner"],
)
q(
    "factual",
    "How frequently does SC-COPPER include a preventive visit?",
    "Every 6 weeks.",
    ["contract_copper"],
)
q(
    "factual",
    "What is the stated warranty period for unopened BP-42 packs?",
    "12 months from dispatch.",
    ["part_warranty"],
)
q("factual", "Which depot dispatches DG-9 shipments?", "The East depot.", ["ship_route"])
q(
    "factual",
    "Where must a rover go after a failed insulation check?",
    "Bay Q2 for quarantine, followed by a new inspection after repair.",
    ["new_fail"],
)
q(
    "factual",
    "Who can release a legal hold on Aster records?",
    "Legal must explicitly release the hold.",
    ["ret_owner"],
)
q(
    "factual",
    "For how long does BAT-L2 remain valid after assessment?",
    "9 months, unless suspended by a failed reassessment.",
    ["train_valid"],
)
q(
    "factual",
    "How many packs did the approved ledger list for BP-42 on 4 February 2026?",
    "36 packs; it is a dated count.",
    ["part_stock"],
)
q("factual", "What was the fiscal 2025 year-end order backlog?", "EUR 1.1 million.", ["fin_orders"])
q(
    "factual",
    "Does printing a battery shipping label prove dispatch?",
    "No. The carrier acceptance scan is proof of dispatch.",
    ["ship_proof"],
)
q("exact_keyword", "Identify the production default for ROVER_SAFE_PATH.", "enabled", ["ops_flag"])
q(
    "exact_keyword",
    "Give the gateway readiness path, distinguishing it from liveness.",
    "GET /ready/rover is readiness; GET /health/rover is liveness.",
    ["ops_probe"],
)
q(
    "exact_keyword",
    "Which paging route reaches the Field Autonomy duty coordinator?",
    "field-auto-primary",
    ["route_page"],
)
q("exact_keyword", "What certification code permits BP-42 handling?", "BAT-L2", ["train_pack"])
q(
    "exact_keyword",
    "Name the exact replacement pack model approved for ORBIT-17.",
    "BP-42, not BP-24.",
    ["part_code"],
)
q("exact_keyword", "What transport classification is assigned to BP-42?", "DG-9", ["ship_class"])
q(
    "exact_keyword",
    "Which service contract code belongs to ORBIT-17 rather than ORBIT-71?",
    "SC-COPPER",
    ["route_unit"],
)
q(
    "exact_keyword",
    "State the quarantine bay identifier used by the current safety rules.",
    "Q2",
    ["new_fail"],
)
q(
    "multihop",
    "What critical acknowledgement time applies to ORBIT-17 under its assigned contract?",
    "25 minutes: ORBIT-17 uses SC-COPPER.",
    ["route_unit", "contract_copper"],
)
q(
    "multihop",
    "Which certificate must a technician hold to handle the approved replacement battery for ORBIT-17?",
    "BAT-L2: its approved battery is BP-42.",
    ["part_code", "train_pack"],
)
q(
    "multihop",
    "Which signed transport document is needed for the replacement battery approved for ORBIT-17?",
    "A signed dangerous-goods declaration: ORBIT-17 uses BP-42, classified DG-9.",
    ["part_code", "ship_class"],
)
q(
    "multihop",
    "On what days can the approved ORBIT-17 replacement pack leave its designated depot?",
    "Tuesdays and Thursdays from the East depot: the pack is BP-42, class DG-9.",
    ["part_code", "ship_class", "ship_route"],
)
q(
    "multihop",
    "Is the approved ORBIT-17 replacement battery automatically paid for by its support contract?",
    "No. It uses SC-COPPER, but contract coverage excludes consumable batteries and requires a separately approved parts order.",
    ["route_unit", "part_code", "contract_charge"],
)
q(
    "multihop",
    "What acknowledgement target and visit interval apply to rover ORBIT-71?",
    "90 minutes and every 12 weeks: ORBIT-71 uses SC-SILVER.",
    ["route_secondary", "contract_silver"],
)
q(
    "multihop",
    "For ORBIT-17, does around-the-clock support imply a guaranteed repair deadline?",
    "No. Its SC-COPPER support operates around the clock but neither contract promises restoration.",
    ["route_unit", "contract_hours", "contract_restoration"],
)
q(
    "multihop",
    "Which production asset battery requires a 9-month handling certificate, and what is that certificate called?",
    "ORBIT-17 uses BP-42; BP-42 handling needs BAT-L2, valid for 9 months.",
    ["part_code", "train_pack", "train_valid"],
)
q(
    "multihop",
    "Why must the price for an ORBIT-17 replacement pack come from the approved catalog rather than the rejected proposal?",
    "The approved pack is BP-42 at EUR 480; the rejected EUR 400 offer never took effect.",
    ["part_code", "part_price", "distract_proposal"],
)
q(
    "multihop",
    "What dispatch classification and preparation lead time apply to the approved replacement pack for ORBIT-17?",
    "It is DG-9; domestic DG-9 orders need 3 working days of preparation after approval before an eligible dispatch day.",
    ["part_code", "ship_class", "ship_lead"],
)
q(
    "comparative",
    "How did the required insulation-check interval change from the archived rules to the 2026 rules?",
    "It shortened from 120 to 80 operating hours.",
    ["old_check", "new_check"],
)
q(
    "comparative",
    "Contrast the return-to-service approval requirements in the 2024 and 2026 safety editions.",
    "2024 allowed the shift supervisor alone; 2026 requires supervisor and certified electrical inspector.",
    ["old_sign", "new_sign"],
)
q(
    "comparative",
    "Contrast the 2024 charging temperature ceiling with the 2026 pause and resume thresholds.",
    "2024 ceiling: 32 Celsius. 2026: pause above 28 and resume only at 26 or below.",
    ["old_temp", "new_temp"],
)
q(
    "comparative",
    "Compare the acknowledgement targets of SC-COPPER and SC-SILVER.",
    "25 minutes for Copper and 90 minutes for Silver.",
    ["contract_copper", "contract_silver"],
)
q(
    "comparative",
    "How does retention differ for routine and incident-linked rover telemetry?",
    "Routine: 45 days after collection. Incident-linked: 18 months after closure.",
    ["ret_telemetry"],
)
q(
    "comparative",
    "How do BP-42 rover packs differ from similarly named warehouse BP-420 cartridges in model suitability and price?",
    "BP-42 costs EUR 480 for ORBIT-17; BP-420 is a EUR 75 trolley cartridge and is unrelated.",
    ["part_code", "part_price", "distract_warehouse", "distract_warehouse2"],
)
q(
    "comparative",
    "Compare the effective and rejected proposed bulk-discount percentages for BP-42.",
    "The approved discount is 15%; the proposed 25% discount was declined.",
    ["part_discount", "distract_proposal2"],
)
q(
    "comparative",
    "Differentiate firmware rollback from clearing a failed insulation check.",
    "Rollback restores the previous firmware but not calibration; a failed insulation check still needs quarantine and a passing post-repair inspection.",
    ["ops_rollback", "new_fail"],
)
q(
    "numerical",
    "Calculate Aster revenue growth from fiscal 2024 to fiscal 2025.",
    "20%.",
    ["fin_revenue"],
    calculation="(3.6 - 3.0) / 3.0 * 100 = 20 percent",
)
q(
    "numerical",
    "What fraction of fiscal 2025 recognized revenue came from service contracts?",
    "60%.",
    ["fin_revenue", "fin_segments"],
    calculation="2.16 / 3.6 * 100 = 60 percent",
)
q(
    "numerical",
    "Compute the fiscal 2025 operating profit margin on recognized revenue.",
    "25%.",
    ["fin_revenue", "fin_profit"],
    calculation="0.9 / 3.6 * 100 = 25 percent",
)
q(
    "numerical",
    "What is the pre-tax total including freight for 10 BP-42 packs under approved terms?",
    "EUR 4115.",
    ["part_price", "part_discount"],
    calculation="10 * 480 * (1 - 0.15) + 35 = 4115 EUR",
)
q(
    "numerical",
    "What is the pre-tax total including freight for an order of 9 BP-42 packs?",
    "EUR 4355; the bulk discount does not apply.",
    ["part_price", "part_discount"],
    calculation="9 * 480 + 35 = 4355 EUR",
)
q(
    "numerical",
    "What is the minimum BAT-L2 passing percentage?",
    "87.5%.",
    ["train_exam"],
    calculation="21 / 24 * 100 = 87.5 percent",
)
q(
    "numerical",
    "By how many minutes is the SC-COPPER acknowledgement target faster than SC-SILVER?",
    "65 minutes.",
    ["contract_copper", "contract_silver"],
    calculation="90 - 25 = 65 minutes",
)
q(
    "numerical",
    "How many cells are contained in a shipment of 12 BP-42 packs?",
    "96 cells.",
    ["part_price"],
    calculation="12 * 8 = 96 cells",
)
q(
    "document_navigation",
    "In the current workshop rules, what fields must appear in the inspection ledger?",
    "Asset ID, operating-hour counter, result and both approvers.",
    ["new_records"],
)
q(
    "document_navigation",
    "Locate the rollout stopping condition and state both its magnitude and duration.",
    "Command rejection above 1.5% for 4 consecutive minutes.",
    ["ops_abort"],
)
q(
    "document_navigation",
    "Within the shipping instructions, what must a DG-9 case label show?",
    "Pack model, dispatch date and handling orientation; never customer passwords.",
    ["ship_label"],
)
q(
    "document_navigation",
    "In the competence register, which date controls an access check?",
    "Certificate expiry, not booking date.",
    ["train_register"],
)
q(
    "document_navigation",
    "Under retention priority, which tag wins when telemetry is both routine and incident-linked?",
    "The incident-linked retention period takes precedence.",
    ["ret_priority"],
)
q(
    "document_navigation",
    "Find the distinction between dispatch preparation and dispatch weekday in the domestic shipment procedure.",
    "Preparation takes 3 working days after approval; dispatch then waits for the next eligible day.",
    ["ship_lead"],
)
q(
    "document_navigation",
    "Which diagnostic artifacts must be captured before resetting the ORBIT-17 motor controller?",
    "The controller event log and battery telemetry bundle.",
    ["route_diag"],
)
q(
    "document_navigation",
    "In the archived rules, how should a November 2025 repair be audited in 2026?",
    "Use the rules effective in November 2025, not the current thresholds.",
    ["old_examples"],
)
q(
    "ambiguous",
    "How long do we keep telemetry?",
    "Clarify whether it is routine or incident-linked: 45 days after collection versus 18 months after closure.",
    ["ret_telemetry"],
    behavior="clarify",
)
q(
    "ambiguous",
    "How quickly will support acknowledge my rover incident?",
    "Ask which contract covers the rover: SC-COPPER is 25 minutes and SC-SILVER is 90 minutes.",
    ["contract_copper", "contract_silver"],
    behavior="clarify",
)
q(
    "ambiguous",
    "What is the battery price?",
    "Ask which model: approved BP-42 packs are EUR 480; warehouse BP-420 cartridges are EUR 75. Quantity affects the BP-42 discount.",
    ["part_price", "part_discount", "distract_warehouse2"],
    behavior="clarify",
)
q(
    "ambiguous",
    "Which safety temperature limit should I apply?",
    "Clarify the date and pause/resume state: the 2024 ceiling was 32 Celsius; current charging pauses above 28 and resumes at 26 or below.",
    ["old_temp", "new_temp", "old_status"],
    behavior="clarify",
)
q(
    "ambiguous",
    "Who signs a return to service?",
    "Ask which rules/date apply: the current edition requires supervisor plus certified electrical inspector, while the archived edition allowed the supervisor alone.",
    ["old_sign", "new_sign", "old_status"],
    behavior="clarify",
)
for prompt, answer in [
    (
        "What is the tax-inclusive cost of one BP-42 pack delivered internationally?",
        "Cannot determine: applicable tax and international shipping costs are not specified.",
    ),
    (
        "Give the personal mobile number of the Field Autonomy duty coordinator.",
        "No personal phone number is provided; use the internal paging route.",
    ),
    (
        "What percentage of Aster revenue came from Japan in fiscal 2025?",
        "No geographic revenue breakdown is disclosed.",
    ),
    (
        "What guaranteed repair completion time applies to SC-COPPER?",
        "No restoration deadline is promised; acknowledgement is not repair completion.",
    ),
    (
        "What secret signing key is installed in the ORBIT-17 controller?",
        "No controller signing key is supplied in the corpus.",
    ),
]:
    q("unanswerable", prompt, answer, behavior="abstain")


# Evaluation: different organisation, document families and investigation chains.
def q(cat, prompt, answer, keys=(), **kw):
    question("held_out", cat, prompt, answer, keys, **kw)


q(
    "factual",
    "At what UTC time do NIGHTGLASS observing blocks begin?",
    "22:30 UTC.",
    ["map_schedule"],
)
q("factual", "Which dome houses LUMEN-8?", "Dome D3.", ["map_room"])
q(
    "factual",
    "From which event is the CCD-K7 manufacturing lead time measured?",
    "Supplier order acceptance; the lead time is 7 weeks and excludes transit.",
    ["detector_lead"],
)
q(
    "factual",
    "Who must approve equipment purchases charged to GR-204?",
    "The research finance officer.",
    ["grant_sign"],
)
q(
    "factual",
    "What happens to unused GR-204 observing hours at quarter end?",
    "They expire and cannot carry forward.",
    alternatives=[["grant_quota", "grant_mirror"]],
)
q(
    "factual",
    "What is the embargo duration for calibrated NIGHTGLASS products?",
    "9 months after acquisition.",
    ["arc_embargo"],
)
q(
    "factual",
    "Which role may reopen an external platform after weather closure?",
    "The facilities controller.",
    ["access_new_after"],
)
q(
    "factual",
    "What is the inspection interval for dome D3 shutter seals?",
    "Every 45 days.",
    ["maint_interval"],
)
q(
    "factual",
    "Where are Meridian Optics detector assemblies dispatched from?",
    "Its North quay laboratory.",
    ["vendor_dispatch"],
)
q(
    "factual",
    "How long is the CCD-K7 warranty, and when does its clock start?",
    "18 months from acceptance testing.",
    ["detector_warranty"],
)
q(
    "exact_keyword",
    "Which archive selector identifies calibrated NIGHTGLASS products?",
    "NG-CAL-6Y",
    ["arc_code"],
)
q(
    "exact_keyword",
    "State the telemetry collector TCP port without substituting the maintenance-console port.",
    "7443, not 7444.",
    ["net_port"],
)
q("exact_keyword", "What value is assigned to OBS_LINK_TTL_SECONDS?", "900 seconds.", ["net_token"])
q(
    "exact_keyword",
    "Which fault code concerns the LUMEN-8 coolant pump?",
    "FLT-208",
    ["maint_fault"],
)
q(
    "exact_keyword",
    "Name the lockout tag applied after a failed coolant isolation check.",
    "COOL-LOCK",
    ["maint_lock"],
)
q(
    "exact_keyword",
    "Which checksum algorithm is verified on archive upload?",
    "SHA-256",
    ["arc_checksum"],
)
q(
    "exact_keyword",
    "Give the queue identifier for routine observatory maintenance.",
    "OBS-MAINT",
    ["maint_queue"],
)
q(
    "multihop",
    "For the programme using LUMEN-8, how many approved quarterly observing hours are available?",
    "160 hours: LUMEN-8 belongs to NIGHTGLASS, funded by GR-204.",
    ["map_instrument", "map_programme"],
    alternatives=[["grant_quota", "grant_mirror"]],
)
q(
    "multihop",
    "Does one replacement detector for LUMEN-8 cross its grant quotation threshold?",
    "Yes. LUMEN-8 uses CCD-K7 at GBP 3200 and NIGHTGLASS uses GR-204, which requires two quotations above GBP 2500.",
    ["map_instrument", "map_programme", "detector_price", "grant_limit"],
)
q(
    "multihop",
    "Where will the approved supplier ship a replacement detector for LUMEN-8 from?",
    "Meridian Optics ships it from the North quay laboratory; LUMEN-8 uses CCD-K7.",
    ["map_instrument", "detector_vendor", "vendor_dispatch"],
)
q(
    "multihop",
    "After how many nights of use must the LUMEN-8 detector receive its calibration exposure sequence?",
    "Every 30 nights: its detector is CCD-K7.",
    ["map_instrument", "detector_maintenance"],
)
q(
    "multihop",
    "Which programme membership is necessary to retrieve embargoed calibrated products from LUMEN-8 observing?",
    "NIGHTGLASS membership: the instrument belongs to NIGHTGLASS and the archive restricts its embargoed products to programme members.",
    ["map_instrument", "arc_membership"],
)
q(
    "multihop",
    "Which seal-inspection interval applies to the dome containing LUMEN-8?",
    "45 days, because LUMEN-8 is in D3.",
    ["map_room", "maint_interval"],
)
q(
    "multihop",
    "Which detector model and supplier should procurement use for the instrument affected by FLT-208?",
    "FLT-208 affects LUMEN-8, which uses CCD-K7 supplied by Meridian Optics.",
    ["maint_fault", "map_instrument", "detector_vendor"],
)
q(
    "multihop",
    "Does being the NIGHTGLASS observing coordinator authorize a GR-204 detector purchase without finance approval?",
    "No. NIGHTGLASS is funded by GR-204; its equipment purchases require research finance officer approval.",
    ["map_programme", "grant_sign"],
)
q(
    "multihop",
    "For a LUMEN-8 detector order, when is the supplier balance payable?",
    "After acceptance testing: LUMEN-8 uses CCD-K7 and Meridian Optics requires the balance after that test.",
    ["map_instrument", "detector_vendor", "vendor_deposit"],
)
q(
    "multihop",
    "Can a visitor host keep the LUMEN-8 external platform operating when sustained wind is 40 km/h in August 2026?",
    "No. LUMEN-8 is in D3; current rules close external platforms above 35 km/h and a booking does not override closure.",
    ["map_room", "access_new_weather"],
)
q(
    "comparative",
    "What changed about visitor escort requirements between the retired and July 2026 editions?",
    "The retired edition allowed unescorted access after induction; the new edition always requires an accredited host.",
    ["access_old", "access_new"],
)
q(
    "comparative",
    "Contrast badge lifetimes under the two visitor-access editions.",
    "72 hours under 2023 rules versus 12 hours under July 2026 rules.",
    ["access_old_hours", "access_new_hours"],
)
q(
    "comparative",
    "How does the external platform wind threshold differ between the retired and current access rules?",
    "It fell from above 45 km/h to above 35 km/h.",
    ["access_old_weather", "access_new_weather"],
)
q(
    "comparative",
    "Distinguish the archive retention clocks for raw frames and calibrated products.",
    "Raw frames: 120 days after acquisition; calibrated products: 6 years after publication.",
    ["arc_raw"],
)
q(
    "comparative",
    "Contrast the instruments LUMEN-8 and LUMEN-80 by detector and programme.",
    "LUMEN-8 uses CCD-K7 for NIGHTGLASS; LUMEN-80 uses CMOS-J2 for DAYFINDER.",
    ["map_instrument", "held_guide"],
)
q(
    "comparative",
    "How does the NIGHTGLASS expansion bid differ from approved terms for quarterly hours and detector price?",
    "The unapproved bid proposed 300 hours and GBP 2800; approved terms are 160 hours and GBP 3200.",
    ["held_draft", "detector_price"],
    alternatives=[["grant_quota", "grant_mirror"]],
)
q(
    "comparative",
    "Are the archive embargo and calibrated-product retention durations measured from the same event?",
    "No. Embargo is 9 months after acquisition; calibrated retention is 6 years after publication.",
    ["arc_embargo", "arc_raw"],
)
q(
    "numerical",
    "What percentage of usable Q2 observing capacity was actually used?",
    "75%.",
    ["cap_hours"],
    calculation="180 / 240 * 100 = 75 percent",
)
q(
    "numerical",
    "How much Q2 instrument budget remained unspent?",
    "GBP 12000.",
    ["cap_spend"],
    calculation="48000 - 36000 = 12000 GBP",
)
q(
    "numerical",
    "Calculate the percentage increase in D3 electricity use from April to May.",
    "20%.",
    ["cap_energy"],
    calculation="(1800 - 1500) / 1500 * 100 = 20 percent",
)
q(
    "numerical",
    "What percentage of Q2 scheduled observing nights were cancelled?",
    "20%.",
    ["cap_nights"],
    calculation="6 / 30 * 100 = 20 percent",
)
q(
    "numerical",
    "How large is the required deposit for two CCD-K7 assemblies before tax and freight?",
    "GBP 1920.",
    ["detector_price", "vendor_deposit"],
    calculation="2 * 3200 * 0.30 = 1920 GBP",
)
q(
    "numerical",
    "How many litres must be added after the documented LUMEN-8 loop drain to restore the full level?",
    "6 litres.",
    ["maint_fluid"],
    calculation="24 - (24 - 6) = 6 litres",
)
q(
    "numerical",
    "Convert the configured archive download-link lifetime to minutes.",
    "15 minutes.",
    ["net_token"],
    calculation="900 / 60 = 15 minutes",
)
q(
    "document_navigation",
    "In the archive procedure, list the three access steps for an embargoed product in order.",
    "Authenticate the researcher, verify programme membership, issue a time-limited link.",
    ["arc_process"],
)
q(
    "document_navigation",
    "Under grant controls, can splitting one equipment purchase into invoices avoid the quotation requirement?",
    "No. Splitting does not remove the requirement.",
    ["grant_limit"],
)
q(
    "document_navigation",
    "Find the step required before any restart for fault FLT-208.",
    "Perform a coolant isolation check.",
    ["maint_fault"],
)
q(
    "document_navigation",
    "What conditions and role are required to release COOL-LOCK?",
    "A certified cooling technician must obtain a passing retest before release.",
    ["maint_lock"],
)
q(
    "document_navigation",
    "What must an order include in addition to its delivery address?",
    "Instrument model and grant code.",
    ["vendor_reference"],
)
q(
    "document_navigation",
    "Locate the telemetry retry profile and give the waits in sequence.",
    "Wait 12 seconds before retry one and 24 before retry two; no further retries.",
    ["net_retry"],
)
q(
    "document_navigation",
    "Which supplier-payment change check must occur before funds are released?",
    "Independent verification by the research finance officer.",
    ["vendor_change"],
)
q(
    "ambiguous",
    "Can I enter the dome without an escort?",
    "Ask when the visit occurs: unescorted access after induction was allowed only under expired rules; current visits require an accredited host.",
    ["access_old", "access_new"],
    behavior="clarify",
)
q(
    "ambiguous",
    "How long is my archive data available?",
    "Clarify raw versus calibrated data and access versus retention: raw lasts 120 days after acquisition, calibrated lasts 6 years after publication, and NIGHTGLASS embargo is 9 months after acquisition.",
    ["arc_raw", "arc_embargo"],
    behavior="clarify",
)
q(
    "ambiguous",
    "Which instrument do you mean by LUMEN?",
    "Request the full identifier. LUMEN-8 is a NIGHTGLASS instrument with CCD-K7; LUMEN-80 is a DAYFINDER guide camera with CMOS-J2.",
    ["map_instrument", "held_guide"],
    behavior="clarify",
)
q(
    "ambiguous",
    "Is it safe to reopen the platform after the weather clears?",
    "Ask whether reopening has been authorized by the facilities controller; a host cannot reopen it solely because weather appears clear.",
    ["access_new_after"],
    behavior="clarify",
)
q(
    "ambiguous",
    "How long do I have to wait for the detector?",
    "Clarify whether manufacturing or full delivery is meant: CCD-K7 manufacturing takes 7 weeks after order acceptance, with unspecified transit time additional.",
    ["detector_lead"],
    behavior="clarify",
)
for prompt, answer in [
    (
        "What is the full landed price including tax and freight for a CCD-K7?",
        "Cannot compute it: the procurement sheet gives neither tax rate nor freight charge.",
    ),
    (
        "Provide the bank account number for a Meridian Optics deposit.",
        "Bank details are withheld; they must come from the finance system.",
    ),
    (
        "Which scientific object was first discovered by LUMEN-8 in Q2?",
        "No discovery results are provided in this corpus.",
    ),
    (
        "How much did the observatory pay for May electricity at dome D3?",
        "Consumption is given but electricity unit prices are absent.",
    ),
    (
        "What are the GR-811 equipment-purchase approval limits?",
        "GR-811 has different conditions, but its purchase limits are not disclosed.",
    ),
]:
    q("unanswerable", prompt, answer, behavior="abstain")


# Markdown tables add structured evidence without indexing question/answer files.
for split, name, key, table in [
    (
        "dev",
        "parts",
        "part_table",
        "| Approved pack | Cells per pack | Case included |\n| --- | --- | --- |\n| BP-42 | 8 | reusable |\n| BP-24 | 6 | reusable |",
    ),
    (
        "held_out",
        "network",
        "net_table",
        "| Public setting | Value | Unit |\n| --- | --- | --- |\n| OBS_LINK_TTL_SECONDS | 900 | seconds |\n| collector listener | 7443 | TCP port |\n| upload limit | 80 | frames |",
    ),
]:
    filename = f"{split}_{name}.md"
    start = len(DOCS[filename])
    DOCS[filename] += table + "\n\n"
    PASSAGES[f"{split}.{key}"] = dict(
        document=filename, split=split, quote=table, char_start=start, char_end=start + len(table)
    )
for item in QUESTIONS:
    if item["question"] == "How many cells are contained in a shipment of 12 BP-42 packs?":
        item["evidence_groups"] = [["dev.part_table"]]
    if item["question"] == "What value is assigned to OBS_LINK_TTL_SECONDS?":
        item["evidence_groups"] = [["held_out.net_token", "held_out.net_table"]]


def build():
    corpus = ROOT / "corpus"
    corpus.mkdir(exist_ok=True)
    for name, text in DOCS.items():
        (corpus / name).write_text(text, encoding="utf-8")
    manifest = {
        name: {
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "split": "held_out" if name.startswith("held_out_") else "dev",
        }
        for name, text in DOCS.items()
    }
    data = {
        "version": "2.0.0",
        "synthetic": True,
        "documents": manifest,
        "passages": PASSAGES,
        "questions": QUESTIONS,
    }
    (ROOT / "benchmark.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    from collections import Counter

    print(
        len(DOCS),
        "documents;",
        len(QUESTIONS),
        "questions;",
        Counter(q["split"] for q in QUESTIONS),
    )
    print(Counter(q["category"] for q in QUESTIONS))


if __name__ == "__main__":
    build()
