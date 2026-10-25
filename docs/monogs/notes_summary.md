**Ba dòng kết luận:**
1. MonoGS (đơn luồng, đã seed) tái lập được paper trên fr1/desk: 1,43 cm so với 1,50 cm. Hai lần chạy cùng seed vẫn lệch vì `atomicAdd` của rasterizer (nhiễu ATE 8 %).
2. E1c giảm ATE keyframe của MonoGS 36–41 % trên snt_far và 65–68 % trên snt_near (vượt nhiễu), nhưng chưa đạt mức "tương tự 2DGS-SLAM": snt_far chỉ giảm 46–48 % ATE mọi frame, dưới ngưỡng 50 %.
3. E1c làm quỹ đạo giật hơn (RPE Δ1 tăng 1,5–2,1 lần). Hạ λ_p xuống 0,1 không đổi ATE (trong nhiễu) và không giảm độ giật.

