#!/system/bin/sh
cd /data/local/tmp
cat sched.pbtxt | perfetto --txt -c - -o - > trace_cpu.pftrace 2> perfetto.log &
PF=$!
cd /data/local/tmp/ncnn && ./benchncnn 100 4 2 -1 1 param=model.param shape=[512,32]
kill -TERM "$PF"        # 优雅停止,perfetto flush trace 到文件;勿用 -9
wait "$PF" 2>/dev/null
