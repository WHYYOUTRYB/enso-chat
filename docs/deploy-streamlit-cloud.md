# 部署 enso-chat 到 Streamlit Community Cloud

> 目标:把对话式 ENSO agent 部署到云端,任何设备浏览器打开公网地址即可演示,不用带自己电脑。
>
> 前提:演示现场有网络。

## 前置准备

1. **GitHub 账号**(免费注册 github.com)
2. **DeepSeek API key**(platform.deepseek.com 获取)
3. 项目代码已提交到本地 git(已完成)

## Step 1: 把项目推到 GitHub

### 1.1 在 GitHub 创建仓库

1. 登录 GitHub,点右上角 `+` → `New repository`
2. 仓库名填 `enso-chat`
3. 选 **Private**(私有,因为含代码)或 Public 都行
4. **不要**勾选 "Add a README"(本地已有)
5. 点 `Create repository`

### 1.2 本地连接远程并推送

在 `D:\作品\enso-chat` 目录打开 Git Bash,执行(把 `你的用户名` 换成你的 GitHub 用户名):

```bash
cd "D:/作品/enso-chat"
git remote add origin https://github.com/你的用户名/enso-chat.git
git branch -M main
git push -u origin main
```

第一次会要求登录 GitHub(用浏览器授权或 personal access token)。

推送后,在 GitHub 网页刷新能看到你的代码文件。

## Step 2: 在 Streamlit Cloud 部署

### 2.1 登录 Streamlit Cloud

1. 打开 https://share.streamlit.io
2. 点 `Continue with GitHub`,用 GitHub 账号授权

### 2.2 创建应用

1. 点 `New app`
2. 配置:
   - **Repository**:选 `你的用户名/enso-chat`
   - **Branch**:`main`
   - **Main file path**:`src/web/app.py`
   - **App URL**(自动生成,可后续自定义)
3. 点 `Advanced settings`:
   - **Python version**:选 3.13(或默认最新)
   - **Secrets**:在文本框粘贴(把你的 key 填进去):
     ```toml
     DEEPSEEK_API_KEY = "sk-你的真实key"
     ```
4. 点 `Save` → `Deploy`

### 2.3 等待部署

- 首次部署约 3-5 分钟(要装 pandas/scikit-learn/streamlit 等依赖)
- 部署日志会实时显示,看到 "Your app is live" 就成了
- 得到公网地址:`https://你的用户名-enso-chat.streamlit.app`

## Step 3: 验证

1. 用任何设备(手机也行)打开那个公网地址
2. 应该看到 "🌊 ENSO 对话式 Agent" 标题
3. 聊天框应该可用(因为 secrets 里配了 key,不会 disabled)
4. 输入 "用示例数据预测 2025 年 1 月的 Niño3.4" 测试

## 演示当天注意事项

### 演示前预热(重要)

Streamlit Cloud 免费版 app **闲置会休眠**。演示前 5-10 分钟先打开网址,等它唤醒(首次加载 1-2 分钟),确保现场不卡。

### 备用方案

- **存一个 key 截图/记事本**:万一 secrets 没生效,可在网页侧栏手动粘贴 key(侧栏输入框仍可用,优先级最高)
- **本地 streamlit 备用**:万一云端挂了,带电脑的话本机 `streamlit run` 应急

### 性能预期

- 云端资源有限,机器学习训练(load_enso_data)比本机慢,约 10-20 秒
- 对话轮次间有 5-15 秒延迟(DeepSeek API + 云端)
- 这些对课程演示可接受,但别承诺"实时"

## 常见问题

### Q: 部署报错 "ModuleNotFoundError"

检查 `requirements.txt` 是否在仓库根目录,且含所有依赖。当前内容:
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
jinja2>=3.1
pytest>=7.4
streamlit>=1.37
```

### Q: 聊天框 disabled(灰色)

说明没读到 key。检查:
1. Streamlit Cloud 的 Secrets 里是否填了 `DEEPSEEK_API_KEY = "sk-..."`
2. 格式必须是 TOML(等号两边空格,key 加引号)
3. 改完 Secrets 要重启 app(在 app 管理页点 "Reboot")

### Q: 首次访问很慢

正常,休眠唤醒。演示前预热即可。

### Q: 想换 key / 撤回

- 换 key:Streamlit Cloud app 设置 → Secrets → 改 → Reboot
- 撤回:删掉整个 app,或把 GitHub 仓库设 Private 后 Streamlit 会停止托管
- **演示完建议轮换 key**(platform.deepseek.com 生成新 key,旧的删掉)

## 安全提醒

- API key 存在 Streamlit Cloud 的 Secrets 里,**不会进 git、不会公开**(只要仓库别设 Public 同时别把 key 写进代码)
- 但 Streamlit Cloud 是第三方服务,key 在它的服务器上——演示完轮换是最稳的
- **绝对不要**把 key 直接写进 `app.py` 或 `config.py` 再推 GitHub(那样 Public 仓库会泄露)
