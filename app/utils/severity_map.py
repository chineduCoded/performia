from app.utils.enums import RiskSeverityLevel, Severity


SEVERITY_MAP = {
    RiskSeverityLevel.LOW: Severity(
        severity_level=RiskSeverityLevel.LOW,
        color="#2ECC71"  # green
    ),
    RiskSeverityLevel.BORDERLINE: Severity(
        severity_level=RiskSeverityLevel.BORDERLINE,
        color="#F1C40F"  # yellow
    ),
    RiskSeverityLevel.MODERATE: Severity(
        severity_level=RiskSeverityLevel.MODERATE,
        color="#E67E22"  # orange
    ),
    RiskSeverityLevel.HIGH: Severity(
        severity_level=RiskSeverityLevel.HIGH,
        color="#E74C3C"  # red
    ),
}
