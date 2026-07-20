# Restricted Claims

Per spec 8.2 and 12.3: the assistant must flag or block the following unless explicitly supported by the approved documents **for the specific product being discussed**. `app/claim_checker.py` implements keyword-based detection of these categories; this file is the human-readable policy it's kept in sync with by hand.

- **ATEX certified** -- not mentioned in any approved document. Never confirm ATEX certification, and never infer it from adjacent-but-different marks (PESO approval, Ex ia IIC T6 Ga, IS/IEC 60079-series testing are not ATEX certification).
- **IECEx certified** -- not mentioned in any approved document. Same restriction as ATEX.
- **SIL / SIL-2 certified** -- not documented for any product in the approved corpus.
- **Life-safety certified** -- not documented for any product.
- **CE marking** -- not mentioned anywhere in the approved documents. Do not confirm or imply CE compliance.
- **Explosion-proof** (as a formal certification claim) -- distinct from the FIXaHY's approved "flameproof" enclosure rating; do not conflate the two.
- **Intrinsically safe** -- only applies to the specific FIXaHY-G/P/E-4220MA-RRNNVVII circuit documented as such. Do not generalize to the whole product line or to the other FIXaHY variant (which has no intrinsic-safety language of its own), and never apply it to AURIGA (explicitly Safe-Area-only).
- **PESO / ARAI / any hazardous-area approval** for a product or configuration not explicitly documented as approved -- e.g. AURIGA is not hazardous-area approved, and neither is the other FIXaHY catalogue variant ("FIXaHY H2 LD XX") despite sharing the FIXaHY name.
- **ppm-level or absolute accuracy claims** not stated in that product's own technical specification table.
- **"Universal" gas detection**, or claiming any specific product detects a gas not explicitly listed for it -- including claiming a currently orderable product can detect helium, methane, SF6, hydrocarbons, or refrigerants. No approved document supports that capability for any current product; every currently orderable product is hydrogen-specific. (Where helium/methane appear at all in the catalogues, it's as a selectable background/interference gas for calibration -- not as a target gas the product detects.)
- **Wireless connectivity (LoRa, Wi-Fi, NB-IoT, Bluetooth)** as a current product feature -- not documented for any product in the approved corpus. Current products document wired interfaces only (RS485, 4-20mA, CAN).
- **Guaranteed delivery date or lead time** -- none is published; do not invent one.
- **Final price, discount, or a warranty term different from the published 12-month standard warranty.**
- **Any regulatory or legal compliance guarantee** beyond what's explicitly documented.
