@echo off
echo ============================================================
echo   AgentRL + 持续学习系统
echo   交互说明：
echo     - 输入自然语言进行文件操作/查询
echo     - 每次回答后可输入 T(有帮助) / F(无帮助) / 回车跳过
echo     - 输入 F 后可选择失败类型进行归因标注
echo     - 输入「学习报告」查看持续学习统计
echo     - 输入「退出」结束程序
echo ============================================================
echo.
"C:\Program Files\Python312\python.exe" -X utf8 "C:\Users\84950\Desktop\AgentRL_IntentRouting\main.py"
pause
