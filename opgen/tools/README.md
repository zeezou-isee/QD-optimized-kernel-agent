# Operator Agent Tools

为 ncnn 算子开发 agent 设计的工具集，灵感来源于 Claude Code 内部工具架构。

## 工具概览

| 工具 | 功能 | 类比 Claude Code |
|------|------|-----------------|
| `read_file` | 读取文件内容（支持文本/图片/PDF） | FileReadTool |
| `write_file` | 创建或覆盖文件 | FileWriteTool |
| `edit_file` | 精确字符串替换编辑 | FileEditTool |
| `grep_search` | 正则搜索文件内容 | GrepTool |
| `glob_search` | 按文件名模式查找文件 | GlobTool |
| `bash_exec` | 执行 bash 命令 | BashTool |

## 工具详情

### 1. read_file — 读取文件

```python
execute_tool("read_file", file_path="/absolute/path/to/file.cpp")
execute_tool("read_file", file_path="/path/to/large.cpp", offset=100, limit=50)
```

- **file_path** (必填): 文件绝对路径
- **offset** (可选): 起始行号（1-indexed），默认 1
- **limit** (可选): 读取行数，默认 2000
- 自动识别图片文件，返回 base64 编码
- 返回带行号的文本内容

### 2. write_file — 写入文件

```python
execute_tool("write_file", file_path="/path/to/new_file.cpp", content="#include <ncnn/mat.h>\n...")
```

- **file_path** (必填): 文件绝对路径
- **content** (必填): 要写入的内容
- 自动创建父目录
- 覆盖已存在的文件
- 返回操作类型（create / update）

### 3. edit_file — 编辑文件

```python
execute_tool("edit_file",
    file_path="/path/to/file.cpp",
    old_string="float* data = new float[size];",
    new_string="auto data = std::make_unique<float[]>(size);")
```

- **file_path** (必填): 文件绝对路径
- **old_string** (必填): 要被替换的精确文本
- **new_string** (必填): 替换后的文本
- **replace_all** (可选): 是否替换所有匹配项，默认 false
- old_string 必须在文件中唯一，否则报错（除非 replace_all=true）

### 4. grep_search — 正则搜索

```python
execute_tool("grep_search", pattern="class\\s+\\w+Layer", glob="*.cpp", output_mode="content", **{"-A": 3})
execute_tool("grep_search", pattern="convolution", path="/path/to/ncnn/src")
```

- **pattern** (必填): 正则表达式
- **path** (可选): 搜索目录/文件，默认 CWD
- **glob** (可选): 文件名过滤（如 `*.cpp`, `*.{h,cpp}`)
- **output_mode** (可选): `content` / `files_with_matches` / `count`
- **-A/-B/-C** (可选): 上下文行数
- **-i** (可选): 大小写不敏感
- **head_limit** (可选): 结果条数上限，默认 250

### 5. glob_search — 文件名搜索

```python
execute_tool("glob_search", pattern="**/*.cpp")
execute_tool("glob_search", pattern="src/layer/**/*conv*")
```

- **pattern** (必填): glob 模式
- **path** (可选): 搜索根目录，默认 CWD
- 结果按修改时间倒序排列
- 最多返回 100 个结果

### 6. bash_exec — 执行命令

```python
execute_tool("bash_exec", command="cmake --build build -j4")
execute_tool("bash_exec", command="python setup.py build", timeout=300000, cwd="/path/to/ncnn")
```

- **command** (必填): 要执行的 bash 命令
- **timeout** (可选): 超时时间（毫秒），默认 120000，最大 600000
- **cwd** (可选): 工作目录
- **env** (可选): 额外的环境变量
- 内置危险命令检测（如 `rm -rf /`）

## LLM 调用方式

### OpenAI API 格式

```python
from operator_agent.tools import TOOL_SCHEMAS, execute_tool

# 1. 将 TOOL_SCHEMAS 传给 LLM
response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    tools=TOOL_SCHEMAS
)

# 2. LLM 返回 function call 后，执行对应的工具
tool_call = response.choices[0].message.tool_calls[0]
result = execute_tool(
    tool_call.function.name,
    **json.loads(tool_call.function.arguments)
)
```

### Anthropic API 格式

```python
from operator_agent.tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

# TOOL_SCHEMAS 已经是 OpenAI 格式，需要转为 Anthropic 格式：
# - type: function -> 直接用 tool name
# - 每个 tool 的 schema 在 function.parameters 中
# 参考 TOOL_MAP 进行调度
```

## 设计原则

1. **简洁统一**: 所有工具遵循相同的返回格式，`{success: bool, ...}` 
2. **安全优先**: bash_exec 内置危险命令检测；文件工具要求绝对路径
3. **权限分离**: 读写操作分离，方便后续接入权限控制
4. **LLM 友好**: 每个工具都有详细的 JSON Schema 描述，方便 LLM 理解参数

## 目录结构

```
operator_agent/tools/
├── __init__.py       # 统一接口（TOOL_SCHEMAS, execute_tool）
├── read_file.py      # 文件读取工具
├── write_file.py     # 文件写入工具
├── edit_file.py      # 文件编辑工具
├── grep_search.py    # 正则搜索工具
├── glob_search.py    # 文件名搜索工具
├── bash_exec.py      # Shell 执行工具
└── README.md         # 本文档
```
