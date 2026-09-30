from __future__ import annotations

from typing import Literal

from pydantic import Field

from tools.contracts import ContractModel

ProfileSection = Literal["overview", "current_diet", "biometrics"]


class ProfileReadInput(ContractModel):
    section: ProfileSection = Field(
        default="overview",
        description=(
            "Use current_diet for the latest logged diet, biometrics for basic current metrics, "
            "or overview only when both are needed."
        ),
    )
