GAP_MODES = [
    {
        "mode":     "D2stDeta",
        "category": r"gap: $D_2^*{\to}D\eta$",
        "label":    r"$B\!\to\!(D_2^*\!\to\! D\eta)\,\ell\nu$",
        "color":    "#e31a1c",
        "ls":       "-",
        "lw":       1.6,
    },
    {
        "mode":     "DummyDeta",
        "category": r"gap: $D\eta$ (non-res.)",
        "label":    r"$B\!\to\! X_{\rm dummy}(\to D\eta)\,\ell\nu$",
        "color":    "#ff7f00",
        "ls":       "--",
        "lw":       1.6,
    },
    {
        "mode":     "Dp1DstEta",
        "category": r"gap: $D_1'{\to}D^*\eta$",
        "label":    r"$B\!\to\!(D_1'\!\to\!D^*\eta)\,\ell\nu$",
        "color":    "#33a02c",
        "ls":       "-",
        "lw":       1.6,
    },
    {
        "mode":     "Dp1Deta",
        "category": r"gap: $D_1'{\to}D\eta$",
        "label":    r"$B\!\to\!(D_1'\!\to\!D\eta)\,\ell\nu$",
        "color":    "#b15928",
        "ls":       "--",
        "lw":       1.6,
    },
    {
        "mode":     "DsKenu",
        "category": r"gap: $D_s K$ (dummy-res.)",
        "label":    r"$B\!\to\! X_{\rm dummy}(\to D_sK)\,\ell\nu$",
        "color":    "#6a3d9a",
        "ls":       "-.",
        "lw":       1.6,
    },
    {
        "mode":     "LcPenu",
        "category": r"gap: $\Lambda_c\bar{p}$ (dummy-res.)",
        "label":    r"$B\!\to\! X_{\rm dummy}(\to\Lambda_c\bar{p})\,\ell\nu$",
        "color":    "#1f78b4",
        "ls":       ":",
        "lw":       1.6,
    },
]

GAP_MODES_BY_NAME = {m["mode"]: m for m in GAP_MODES}
