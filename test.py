from dataclasses import dataclass


@dataclass(frozen=True)
class Solution:
    f: float


sol = Solution(9.0)

quit()
