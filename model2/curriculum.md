# Static curriculum

One teacher scores the opposite dataset using its synthetic answers. Freeze each image difficulty and easy/medium/hard membership before the student's training. Model2 changes the sampling mixture according to a predetermined schedule; the ranking itself stays fixed.

Historical schedule: epoch1 easy-only; epoch2 50%easy/50%medium; epochs3-20 15%easy/25%medium/60%hard. Historical scoring used a three-reader ensemble and95k/5k split. New A/B runs will use one opposite-dataset teacher and100k optimization images; adaptation and final schedule confirmation are pending.

Store each new initial ranking once under rankings/darkbay/A or B. Model3 references these same files rather than duplicating them. Real labels play no role in difficulty.
