# KAPPA_HORIZON — độ cứng render và sai số theo tầm (PLAN MONOGS-PW2 §4.6, chỉ báo)

Sinh bằng `tools/monogs/pw2/kappa_horizon.py` trên hai run `base` hiệu chỉnh của R1 (frame 0–300, đo độ cứng mỗi 5 frame). ε_Δ = Log(M_est·M_gt⁻¹), M = T(t)·T(t−Δ)⁻¹, theo trục τ (m, rad).

## snt_far_val

| Đại lượng | tx | ty | tz | rx | ry | rz |
|---|---|---|---|---|---|---|
| trung vị c_render | 12,3 | 18,5 | 4,48 | 50,8 | 44,1 | 11,5 |
| trung vị ε² (Δ = 1) | 8,05e-06 | 3,36e-06 | 7,99e-07 | 1,07e-06 | 2,82e-06 | 1,8e-06 |
| Spearman(c_render, ε²) | -0,14 | -0,38 | -0,35 | -0,11 | -0,3 | -0,31 |
| trung vị(ε_Δ²·c)/Δ, Δ = 1 | 0,00012 | 5,27e-05 | 3,9e-06 | 4,88e-05 | 0,000104 | 1,97e-05 |
| trung vị(ε_Δ²·c)/Δ, Δ = 5 | 0,00039 | 4,67e-05 | 6,32e-06 | 4,16e-05 | 0,000131 | 1,35e-05 |
| trung vị(ε_Δ²·c)/Δ, Δ = 20 | 0,000567 | 1,64e-05 | 1,72e-05 | 1,29e-05 | 0,000381 | 2,06e-05 |
| trung vị(ε_Δ²·c)/Δ, Δ = kf | 0,000187 | 5,58e-05 | 9,72e-06 | 1,81e-05 | 0,000192 | 2,35e-05 |

## st_far_val

| Đại lượng | tx | ty | tz | rx | ry | rz |
|---|---|---|---|---|---|---|
| trung vị c_render | 281 | 292 | 58,1 | 728 | 671 | 261 |
| trung vị ε² (Δ = 1) | 8,47e-06 | 3,82e-05 | 1,26e-05 | 1,72e-05 | 2,49e-06 | 2,66e-06 |
| Spearman(c_render, ε²) | -0,3 | -0,16 | -0,25 | -0,063 | -0,043 | -0,0084 |
| trung vị(ε_Δ²·c)/Δ, Δ = 1 | 0,00193 | 0,0105 | 0,000588 | 0,0132 | 0,00157 | 0,000847 |
| trung vị(ε_Δ²·c)/Δ, Δ = 5 | 0,000451 | 0,0026 | 0,000189 | 0,00301 | 0,000806 | 0,000117 |
| trung vị(ε_Δ²·c)/Δ, Δ = 20 | 0,000257 | 0,00067 | 3,27e-05 | 0,00107 | 0,000222 | 6,8e-05 |
| trung vị(ε_Δ²·c)/Δ, Δ = kf | 0,000361 | 0,000596 | 2,01e-05 | 0,00125 | 0,000307 | 8,26e-05 |

**Đọc:** nếu dòng trung vị(ε_Δ²·c)/Δ tăng mạnh theo Δ, sai số render tương quan theo thời gian (trôi); khi đó κ đo từ hai frame liền nhau làm render được tin quá mức về lâu dài. Spearman dương nghĩa là trục render cứng hơn lại có sai số lớn hơn, ngược với giả định của 1b.

