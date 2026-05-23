"""Published HiFD constants and application profiles.

These are the *defaults* shipped with the package. Override by passing your
own plain dict to any API that accepts ``constants=`` / ``profiles=``, or via
``--constants FILE`` / ``--profiles FILE`` on the CLI.
"""

from types import MappingProxyType

SCHEMA_VERSION = "1.0"

DEFAULT_CONSTANTS = MappingProxyType({
    "Delta_age":     100.0,
    "tau_NME":       0.10,
    "theta_max_deg": 90.0,
    "tau_HR":        20.0,
    "tau_N":         5.0,
    "alpha":         0.5,
    "epsilon":       1.0e-6,
})

DEFAULT_PROFILES = MappingProxyType({
    "privacy_first": MappingProxyType(
        {"P": 0.50, "Q": 0.125, "U1": 0.125, "U2": 0.125, "U3": 0.125}
    ),
    "balanced": MappingProxyType(
        {"P": 0.20, "Q": 0.20, "U1": 0.20, "U2": 0.20, "U3": 0.20}
    ),
    "clinical": MappingProxyType(
        {"P": 0.15, "Q": 0.15, "U1": 0.15, "U2": 0.15, "U3": 0.40}
    ),
})
