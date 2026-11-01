# 重启后恢复清单（2026-11-01 wedge 之后）

## 1. 删掉临时 drop-in（它禁用了服务自动重启）
```
rm -f ~/.config/systemd/user/dsv41.service.d/norestart.conf
systemctl --user daemon-reload
```

## 2. 确认显卡回来了
```
source /opt/intel/oneapi/setvars.sh --force
sycl-ls | grep B70        # 必须能看到 Arc(TM) Pro B70
```

## 3. 启动服务并验证
```
systemctl --user start dsv41.service
sleep 5; systemctl --user is-active dsv41.service
cd ~/deepseek-work/llm-for-glm-flash && python3 bench.py   # 期望冷跑 ~19 tok/s，第二个 ~23-24
```

## 4. 本轮已落地、需要记住的状态
- **生产配置未改**：pack-real / experts 6 / vram-experts 0.9 / dspark off / 384K / -O3。
- 源码里新增的插桩**全部由环境变量门控、默认关**：`DS4_PROF`、`DS4_PORT_LAYER`、
  `DS4_WIN_HOSTWAIT`（窗口路径主机侧等待，实测 +1.3% 但没解决缺口，保持关闭）。
- MTP 结论：窗口缺口 ~65 ms 与专家数无关，即使全修好也只到 24.9 tok/s ≈ 基线，**这条线关闭**。
- 两个上游 issue 已挂（非 PR）：Niko1221/Strata#4919、Headmaster218/MoE4All#151。
- poorseek main 已改为"单配置框架"口径（f44694f）。

## 5. 新硬规矩（这次 wedge 换来）
- 手动引擎跑完后**等 60–90 s** 再启服务；
- 任何一次引擎启动失败（OOM / DEVICE_LOST）→ **立刻 stop 服务 + 加 `Restart=no` drop-in**，
  别让 `Restart=always` 每 5 秒去碰一块正在复位的卡。
