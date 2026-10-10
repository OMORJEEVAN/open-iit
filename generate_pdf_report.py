import os
import sys

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)

def create_report():
    BASE_DIR = os.getcwd()
    pdf_filename = "CreditNirvana_PS2_Project_Report.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceAfter=15,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=12,
        spaceAfter=6,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#2D3748"),
        spaceBefore=8,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2D3748"),
        leftIndent=15,
        spaceAfter=4,
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1A202C"),
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1A365D"),
    )

    story = []

    # Title & Metadata Header
    story.append(Paragraph("CreditNirvana Product & Research Challenge", subtitle_style))
    story.append(Paragraph("Problem Statement 2: Right-Party Contact (RPC) Prediction & Skip-Trace Prioritisation (Tele & Field)", title_style))
    story.append(Paragraph("<b>Domain:</b> AI-Native Debt Collections & Recovery | Retail, MSME & Microfinance Portfolios in India", body_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=12))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary & Benchmark Impact", h1_style))
    story.append(Paragraph(
        "In debt collections, incumbent diallers and field agencies waste over 60% of their operational budget on dead contact points or unproductive redialling loops on avoiding borrowers. Meanwhile, skip-tracing is traditionally triggered by arbitrary fixed-attempt heuristics (e.g., 15 failed calls), incurring high vendor costs (₹89/trace) with minimal recovery realization.<br/><br/>"
        "This project implements an autonomous, end-to-end collections optimization system that decouples <b>Line Liveness P(Active)</b> from <b>Borrower Responsiveness P(RPC | Active)</b>, operationalizes both <b>Telephony and Field</b> channels, corrects historical policy bias via Inverse Propensity Weighting (IPW), and economically optimizes skip-tracing through an Expected Value-of-Information (VoI) framework.",
        body_style
    ))

    # Benchmark Table
    metrics_data = [
        [Paragraph("<b>Official Competition Metric (Page 7)</b>", table_header_style),
         Paragraph("<b>Incumbent Baseline Rule</b>", table_header_style),
         Paragraph("<b>Model Policy Engine</b>", table_header_style),
         Paragraph("<b>Operational Impact / Lift</b>", table_header_style)],
        [Paragraph("Right-Party Contact (RPC) Rate / 1,000 Dials", table_cell_bold),
         Paragraph("164.4 RPCs / 1,000", table_cell_style),
         Paragraph("<b>227.1 RPCs / 1,000</b>", table_cell_bold),
         Paragraph("<b>+38.2% Lift</b> in connect productivity", table_cell_style)],
        [Paragraph("Model Probability Calibration (Brier Score)", table_cell_bold),
         Paragraph("N/A (Uncalibrated heuristic)", table_cell_style),
         Paragraph("<b>0.0789 (Liveness)</b> / 0.1338 (RPC)", table_cell_bold),
         Paragraph("Calibrated for live credit decisioning", table_cell_style)],
        [Paragraph("Model Discrimination (Out-of-Sample ROC-AUC)", table_cell_bold),
         Paragraph("N/A", table_cell_style),
         Paragraph("<b>0.7771 (Liveness)</b> / 0.7140 (RPC)", table_cell_bold),
         Paragraph("Disentangles liveness from avoidance", table_cell_style)],
        [Paragraph("Dial Attempts Wasted on Dead Lines Before Cutoff", table_cell_bold),
         Paragraph("15.0 calls / contact", table_cell_style),
         Paragraph("<b>5.6 calls / contact</b>", table_cell_bold),
         Paragraph("<b>-62.4% Reduction</b> in wasted dialler overhead", table_cell_style)],
        [Paragraph("Skip-Trace Queue Hit Rate", table_cell_bold),
         Paragraph("22.8% (15-call rule)", table_cell_style),
         Paragraph("<b>24.5% (VoI Ranker)</b>", table_cell_bold),
         Paragraph("Screened across tele & field channels", table_cell_style)],
        [Paragraph("Skip-Trace Projected Net Recovery Value", table_cell_bold),
         Paragraph("Marginal / Negative ROI", table_cell_style),
         Paragraph("<b>INR 11,946,738.33</b>", table_cell_bold),
         Paragraph("Prioritizes high-balance, reachable debt", table_cell_style)],
        [Paragraph("Third-Party Debt Disclosure Incidents (RBI FPC)", table_cell_bold),
         Paragraph("Vulnerable under legacy dialler", table_cell_style),
         Paragraph("<b>0 Incidents (Target: ZERO)</b>", table_cell_bold),
         Paragraph("1,635 contacts proactively suppressed", table_cell_style)],
    ]

    t_metrics = Table(metrics_data, colWidths=[165, 110, 115, 140])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1A365D")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 10))

    # Core Modeling Dilemmas Solved
    story.append(Paragraph("2. Resolving the Central Modelling Dilemmas", h1_style))
    story.append(Paragraph("<b>Problem A: Borrower Avoiding vs. Dead Number</b>", h2_style))
    story.append(Paragraph(
        "A borrower deliberately avoiding calls and an unassigned/dead SIM appear identical on the surface (both result in 0 right-party conversations and repeated failures). However, their operational responses are diametrically opposite: <b>Switch Channel</b> (WhatsApp, field visit) for the first, and <b>Skip-Trace</b> for the second.<br/>"
        "We decouple observation into two latent probabilities: <i>Observed Connect = P(Active Line) × P(RPC | Active Line)</i>. Using telephony physics (customer decline events where <code>hangup_by == 'customer'</code>, full 30–45s ring duration vs 0–2s instant circuit drops) and cross-channel field visit confirmation (<code>locked_premises</code>, <code>met_family</code>), we separate the two: <b>3,478 attempts (47.0%)</b> are identified as avoiding borrowers and routed to digital/field channels, while <b>379 dead lines (5.1%)</b> are flagged for skip-trace.",
        body_style
    ))

    story.append(Paragraph("<b>Problem B: Switched Off Long-Term vs. Temporarily Unreachable</b>", h2_style))
    story.append(Paragraph(
        "An isolated <code>switched_off</code> event often signifies traveling or a dead battery (<b>Temporarily unreachable</b> → retry later with backoff). However, persistent switch-offs across multiple days and diurnal slots indicate discarded SIMs (<b>Switched off long-term</b> → move to another number on file, or trace).<br/>"
        "By tracking unbroken streak length across diverse time windows, the model identifies <b>215 persistent switch-offs</b>. Crucially, the policy engine cascades <b>171 attempts (2.3%)</b> to <b>Move to another number on file</b> because an alternative phone existed in <code>phones.csv</code>, reserving skip-trace costs exclusively for exhausted accounts.",
        body_style
    ))

    # Page Break for clean visual layout
    story.append(PageBreak())

    # Full Tele & Field Scope
    story.append(Paragraph("3. Multi-Channel Scope: Telephony & Field Decisions", h1_style))
    story.append(Paragraph(
        "The problem statement explicitly spans both Tele and Field collections. Our system provides complete state-to-action mapping for both channels according to the Problem Statement 2 table:",
        body_style
    ))

    story.append(Paragraph("<b>Telephony Channel Prescribed Actions (Test Set - 7,407 Dials)</b>", h2_style))
    story.append(Paragraph("• <b>Switch channel (WhatsApp, field) instead of redialling:</b> 3,478 attempts (47.0%) — Halts futile calls on avoiding borrowers.<br/>"
                           "• <b>Keep dialling, at best time slot:</b> 1,299 attempts (17.5%) — Targets reachable borrowers at their highest-probability diurnal slot.<br/>"
                           "• <b>Stop at once (Suppression):</b> 877 attempts (11.8%) — Recycled numbers suppressed to guarantee 0 debt disclosure to strangers.<br/>"
                           "• <b>Use within Fair Practices Code rules:</b> 758 attempts (10.2%) — Reference and employer contacts locked under FPC compliance.<br/>"
                           "• <b>Trigger Skip-Trace:</b> 423 attempts (5.7%) — Targets confirmed dead lines and exhausted accounts.<br/>"
                           "• <b>Retry later, with backoff:</b> 401 attempts (5.4%) — Transient network unavailability / isolated switch-offs.<br/>"
                           "• <b>Move to another number on file:</b> 171 attempts (2.3%) — Cascades accounts with dead primary numbers to backup lines.", bullet_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Field Operations Channel Prescribed Actions (3,117 Addresses Evaluated)</b>", h2_style))
    story.append(Paragraph("• <b>Visit:</b> 2,295 addresses (73.6%) — Valid and occupied residences confirmed by prior contact or complete KYC description.<br/>"
                           "• <b>Change the visit time:</b> 478 addresses (15.3%) — Locked premises or family met; shifts visit to evening/weekend windows.<br/>"
                           "• <b>Resolve the location (PS3 Geocoder):</b> 233 addresses (7.5%) — Detailed address description but agent could not locate pin.<br/>"
                           "• <b>Trace the new address:</b> 110 addresses (3.5%) — Neighbours confirmed borrower relocated; triggers skip-trace.<br/>"
                           "• <b>Trace, and flag to origination team:</b> 1 address (<0.1%) — Incomplete or fabricated origination address.", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("4. Visual Performance Diagnostics", h1_style))

    # Add images side by side if available
    plots_dir = os.path.join(BASE_DIR, "output", "plots")
    p1 = os.path.join(plots_dir, "plot1_calibration_curves.png")
    p2 = os.path.join(plots_dir, "plot2_roc_curves.png")
    p3 = os.path.join(plots_dir, "plot3_rpc_rate_comparison.png")
    p4 = os.path.join(plots_dir, "plot4_wasted_attempts_reduction.png")
    p5 = os.path.join(plots_dir, "plot5_action_distribution.png")

    if os.path.exists(p1) and os.path.exists(p2):
        img_table = Table([[
            Image(p1, width=255, height=105),
            Image(p2, width=255, height=105),
        ]])
        img_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
        story.append(img_table)
        story.append(Paragraph("<i>Figure 1: Probability Calibration Curves (Reliability Diagrams) & Out-of-Sample ROC Curves</i>", ParagraphStyle("Cap", parent=body_style, fontSize=7.5, alignment=1, textColor=colors.gray)))

    if os.path.exists(p3) and os.path.exists(p4):
        story.append(Spacer(1, 6))
        img_table2 = Table([[
            Image(p3, width=255, height=105),
            Image(p4, width=255, height=105),
        ]])
        img_table2.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
        story.append(img_table2)
        story.append(Paragraph("<i>Figure 2: Operational Benchmark Lift (+38.2% RPC Rate & -62.4% Wasted Attempts Reduction)</i>", ParagraphStyle("Cap2", parent=body_style, fontSize=7.5, alignment=1, textColor=colors.gray)))

    # Page Break for clean conclusion and regulatory compliance
    story.append(PageBreak())

    story.append(Paragraph("5. Value-of-Information (VoI) Skip-Trace Optimization", h1_style))
    story.append(Paragraph(
        "Traditional collections policies trigger skip-tracing after a fixed count (e.g. 15 consecutive failed calls), wasting ₹89 on accounts with uncollectible balances or accounts where the borrower is simply avoiding and reachable via field visits.<br/>"
        "We replace this with an economic Value-of-Information (VoI) formulation:<br/>"
        "<b>E[Net Recovery] = P(Hit) × P(RPC | Found) × (Outstanding × Ability-to-Pay × Realization Rate) - Cost(₹89)</b><br/><br/>"
        "Multi-channel screening verifies that if an account has an active phone line, an avoiding phone where WhatsApp/field can be used, or a confirmed residential address, it is <b>disqualified from burning skip-trace capital</b>. This narrows the queue down to the truly untraceable, high-value accounts, saving the lender over ₹1.7 Lakhs in vendor fees.",
        body_style
    ))

    story.append(Paragraph("6. Ground Truth Audit & Regulatory Compliance", h1_style))
    story.append(Paragraph("<b>Post-Campaign Ground Truth Audit (250 Contacts):</b>", h2_style))
    story.append(Paragraph(
        "Benchmarking against <code>verified_contact_points.csv</code> confirmed that <b>zero verified borrower numbers</b> were misclassified as dead or invalid. Furthermore, 100% of recycled and third-party numbers were filtered away from direct collections dialling, ensuring zero risk of stranger debt disclosure.",
        body_style
    ))

    story.append(Paragraph("<b>Regulatory Adherence (RBI Fair Practices Code & DPDP Act):</b>", h2_style))
    story.append(Paragraph(
        "• <b>Explainability & Auditability:</b> Every prediction record in <code>test_predictions_with_policy.csv</code> contains an automated, deterministic <code>action_reason</code> code for supervisors and RBI auditors.<br/>"
        "• <b>Anti-Harassment Guardrails:</b> Avoiding borrowers are never subjected to high-frequency dialler bombardment; they are transitioned to respectful digital reminders and structured restructuring offers.<br/>"
        "• <b>Continuous Feedback Loop via Exploration:</b> To prevent data-starvation feedback loops (PS2 Page 6), 95% of traffic exploits model scores while 5% is allocated to a randomized exploration budget with logged propensities to continuously refresh contact states.",
        body_style
    ))

    story.append(Paragraph("7. Submission Deliverables Summary", h1_style))
    story.append(Paragraph(
        "• <b>Jupyter Notebook:</b> <code>CreditNirvana_PS2_Solution.ipynb</code> (Single self-contained notebook, fully executed with embedded tables and plots).<br/>"
        "• <b>Executive Project Report:</b> <code>CreditNirvana_PS2_Project_Report.pdf</code> (This official report document).<br/>"
        "• <b>Output Decision Artifacts:</b> All scored CSV files and priority queues available in <code>output/</code>.",
        body_style
    ))

    doc.build(story)
    print(f"PDF Report generated successfully: {pdf_filename}")

if __name__ == "__main__":
    create_report()
