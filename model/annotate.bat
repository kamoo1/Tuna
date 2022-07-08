:: D:\opencv\build\x64\vc15\bin\opencv_annotation.exe --annotations=pos.txt --images=pos/
:: D:\opencv\build\x64\vc15\bin\opencv_createsamples.exe -info pos.txt -w 24 -h 24 -num 300 -vec pos.vec
:: env.bat
:: 133
D:\opencv\build\x64\vc15\bin\opencv_traincascade.exe -data dist/ -vec pos.vec -bg neg.txt -w 24 -h 24 -numPos 100 -numNeg 300 -numStages 9 -maxFalseAlarmRate 0.3 -minHitRate 0.99 > train.log
