# Function Calling 与 MCP 协议｜深究 MCP 协议的设计

![](images/0b4fd33a020e5a5ff8ed68dbc8e607d1364f762b58cdf260462618c6344f28b0.jpg)

该⽂档在线链接和 PDF 版本均在视频简介。

## ⼀、Function Calling

## 2.1要解决的问题

传统聊天⼤模型只会说话，没有⼯具调⽤能⼒，这使得⼤模型：

1. ⽆法感知环境：⽆法与外部数据源交互，如通过API查询⽹⻚、查看⽤⼾本地⽂件、访问远程数据库等等

2. ⽆法改变环境：⽆法帮⽤⼾实际执⾏任务，如跑代码、发邮件、上传作业等

2.2如何解决问题

后端 + LLM

传统⽅案

⼯作流程

![](images/56c220acff0046bf9daf82dbaf2f093455456e3d25a6176122f37fe6d4d586a5.jpg)  
Hard Code ⽅案 ｜ by @bilibili 堂吉诃德拉曼查的英豪（Carbon Based）

## 存在的问题

1. 是否调⽤⼯具、调⽤什么⼯具由后端负责判断，逻辑复杂且容易误判。

![](images/8c705d03c695ba6c47faeffa5e7915d9480a35c0e726bae512923f3b622e9946.jpg)

AI 这么智能，为什么不让它来帮我判断？

2. 调⽤⼯具的参数由后端负责构建，难度很⼤。

![](images/1caecf90bec0cd37a6911bc7982e6e8568cf3f1deaf325f36f73a1049350cb1a.jpg)

AI 这么智能，为什么不让它来帮我⽣成参数？

## Function Calling ⽅案

Function Calling 是什么

⼴义的FunctionCalling是指让⼤模型能够调⽤外部⼯具的⼀种技术实现：先向⼤模型提供可⽤函数的列表及说明，由⼤模型在对话过程中智能判断是否需要调⽤函数，并⾃动⽣成调⽤所需的参数，最终⽤⽂字返回符合约定格式的函数调⽤请求。

狭义的 Function Calling 特指⼤模型提供商在模型内部与 API 层⾯做了⽀持的⼀种能⼒，它最早由OpenAI 引⼊：

在模型层⾯：模型提供商需对⼤模型进⾏特别优化，使其具备根据上下⽂正确选择合适函数、⽣成有效参数的能⼒（⽐如有监督微调、强化学习）。

• 在API层⾯：模型提供商需额外开放对FunctionCalling的⽀持（⽐如GPTAPI 中提供了⼀个functions 参数）。

## 基于提⽰词的 Function Calling

⼯作流程  
![](images/f453795f397758062e3eef80312b3e96e7807c660036d0b5723bde9843fa5f75.jpg)

基于提⽰词的⽅案 ｜ by @bilibili 堂吉诃德拉曼查的英豪（Not Ai，Carbon Based Version）

System Prompt ⽰例  
1  
2 你是⼀个函数调⽤助⼿，我将提供多个函数的定义信息，包括函数名称、作⽤、参数及参数类型。  
3  
4  
5 - 根据⽤⼾的输⼊，判断是否需要调⽤某个函数  
6 - 如果需要，请\*\*严格按照以下格式\*\*输出函数调⽤指令：  
7 \`\`\`json  
8 { "name": "函数名", "arguments": { "参数名": "参数值" } }  
9  
10  
11  
12 1. \*\*get_weather\*\*  
13 - 作⽤：查询指定城市的天⽓情况

14 - 参数：  
15 -\`city\`（string）：城市名称

2. \*\*get_time\*\*16

17 - 作⽤：查询指定城市的当前时间

18 - 参数：

19 - \`city\`（string）：城市名称

⽤⼾提问⽰例

“⼴州的天⽓怎么样？”1

模型回复⽰例

{ "name": "get_weather", "arguments": { "city": "⼴州" } 1

## 存在的问题

1. 输出格式不稳定。如调⽤指令中存在多余⾃然语⾔。

2. 容易出现幻觉。模型可能编造并不存在的函数名或参数。

?? ⼤模型提供商能否对模型进⾏微调、强化学习，提升⼤模型在这⼀⽅⾯的能⼒？

3. 对开发者依赖度⾼。函数描述、调⽤指令格式、提⽰词逻辑完全由开发者设计。

函数描述、调⽤指令格式能否由⼤模型提供商来指定？系统提⽰词中的“说明与规则”逻辑能否由⼤模型提供商来兜底？

4. 上下⽂冗⻓，Token 消耗⼤。为确保调⽤逻辑正确，往往需要在 system prompt 中加⼊⼤量说明与规则。

基于 API 的 Function Calling

⼯作流程

![](images/f7692bcf06e96518c6bc1ce8b6baf56abfd56313f897728cf874016f985da240.jpg)  
基于 API 的 Function Calling ｜ by @bilibili 堂吉诃德拉曼查的英豪（Not Ai，Carbon Based Version）

## 1. ⽤⼾发起提问

⽤⼾通过⾃然语⾔提出问题，例如：“⼴州今天天⽓如何？适合出⻔吗？

## 2. 后端第⼀次向⼤模型API发起请求，获取函数调⽤指令

后端向⼤模型API传⼊⽤⼾原始输⼊、 函数描述和其他上下⽂信息，获取调⽤指令。函数描述包括函数名称、⽤途说明、参数结构等。

接⼝⼊参  
1 {  
2 "messages": [  
3 {  
4 "role": "system",  
5 "content": "你是⼀个助⼿，可以根据⽤⼾的请求调⽤⼯具来获取信息。 II  
6 },  
7 {  
8 "role": "user",  
9 "content": "⼴州今天天⽓如何？适合出⻔吗？"  
10 }  
11 ],  
12 "functions": [  
13 {  
14 "name": "getWeather",  
15 "description": "获取指定城市的天⽓",

16 "parameters": {   
17 "type": "object",   
18 "properties": {   
19 "location": {   
20 "type": "string",   
21 "description": "城市名称，⽐如北京"   
22 },   
23 "date": {   
24 "type": "string",   
25 "description": "⽇期，⽐如 2025-08-07"   
26 }   
27 },   
28 "required": ["location", "date"]   
29 }   
30 }   
31 ]   
32 }   
33

## 3. 模型⽣成调⽤指令

模型会智能判断是否需要调⽤函数，选择合适的函数，并基于上下⽂⾃动⽣成结构化的调⽤指令（函数名+参数），例如：

调⽤指令⽰例 （OpenAI Function Calling）   
1 {   
2 "function_call": {   
3 "name": "getWeather",   
4 "arguments": {   
5 "location": "Guangzhou",   
6 "date": "2025-07-17"   
7 }   
8 }   
9 }

## 4. 后端解析调⽤指令，并执⾏实际的函数调⽤

后端接收到模型返回的调⽤指令后，解析调⽤指令，得到函数名称和参数，执⾏对应的⽅法（如调⽤天⽓查询函数），并获取结果。调⽤指令例如：

## 5. 后端第⼆次向⼤模型 API 发起请求，将刚才的调⽤结果和其他上下⽂信息⼀起传给模型，⽣成最终的回复

后端将函数执⾏结果+其他上下⽂信息（包括⽤⼾原始输⼊）传给模型，模型判断此时已有⾜够的信息回答问题，不再需要调⽤函数了，于是直接⽣成最终结果，例如：“⼴州今天35度，暴⾬，建议在室内活动”。

## 存在的问题

1. 后端应⽤适配不同⼤模型时存在⼤量冗余开发

2. 可选模型有限

## ⼆、MCP 协议

## 3.1要解决的问题

## 1. ⼯具接⼊的冗余开发问题

AI 应⽤接⼊他⼈开发的新⼯具需完整 copy 代码和函数描述，接⼊⼏个就要 copy ⼏次。

## 2. ⼯具复⽤困难

环境问题导致 copy 的代码不⼀定能跑；很多企业不提供可供 copy 的源码；跨语⾔的代码 copy了没⽤。

## 3.2如何解决问题

如果是你会怎么解决以上问题？（假设现在完全没有 MCP 协议）

## 从问题出发

-> 冗余开发、复⽤困难问题基本都是由 copy 代码带来的，怎么才能⽤上别⼈的⽅法，但⼜不⽤ copy别⼈的代码？

• 思路⼀：导包式接⼊，AI 应⽤开发者将⼯具代码拉到本地调⽤。

◦ 可⾏吗？⚠️跨语⾔调⽤问题⽆法解决

◦ 跨语⾔问题是客观⽆解的吗？->能否本地另起⼀个进程运⾏该语⾔的执⾏环境，⼯具代码原封不动运⾏在对应语⾔环境中，AI 应⽤再通过进程间通信（如管道、套接字）获取标准化的返回结果？

◦ 所以可⾏吗？✅->并且，这种接⼊⽅式更适合被称为“本地服务式接⼊”

• 思路⼆：远程服务式接⼊，⼯具开发者将⼯具独⽴部署，封装成标准化API，约好统⼀的请求/响应格式，AI应⽤开发者只需按规定传⼊参数并解析返回值即可，不需要关⼼⼯具的实现语⾔、运⾏环境、内部逻辑。

◦ 可⾏吗？

## 从⽬标出发

->接⼊⼯具、复⽤⼯具对开发者来说最理想的⽅式是什么？

• 开发者只需添加⼀条配置（如⼯具的唯⼀标识或访问地址）就可以接⼊⾃⼰ / 别⼈的⼯具

## ->当前的⾮理想状态是什么？

• 当前每新增⼀个⼯具，开发者需要在链路中做两处⼈⼯适配：

a. 补充⼯具描述

b. 补充⼯具代码

## ->从⾮理想状态到理想状态必须满⾜什么条件？

• 要让“配置替代⼈⼯适配”，必须满⾜以下两点：

◦ 新增配置后，AI 应⽤后端要能根据配置⾃动获取⼯具的描述信息

◦ 新增配置后，AI应⽤后端要能根据配置⾃动定位⼯具调⽤⼊⼝并执⾏调⽤

## 从问题到技术需求

## -> 本地服务式接⼊场景

• 如何在任意 AI 应⽤中通过⼀条标准化配置：

◦ 拉取任意⼯具的包到本地，并在本地起⼀个进程，将⼯具作为⼀个服务运⾏起来（只要⼯具开发者有提供对应的包）

◦ ⾃动获取⼯具的描述信息、⾃动完成调⽤过程（通过本地进程间通信）

## ->远程服务式接⼊场景

• 如何在任意 AI应⽤中通过⼀条标准化配置：

◦ 访问任意⼯具的远程服务（只要⼯具开发者有对外提供服务）

◦ ⾃动获取⼯具的描述信息、⾃动完成调⽤过程（通过远程服务调⽤）

## 技术⽅案思路

1. ⼯具与AI应⽤必须解耦合。

2. ⼯具与AI应⽤之间的交互必须标准化。

a. AI 应⽤和⼯具服务的通信协议需要统 （本地进程间的通信协议/远程服务调⽤的协议）b. AI 应⽤和⼯具服务的接⼝定义需要统 （需要提供哪些接⼝、接⼝需包含哪些参数）c. AI 应⽤和⼯具服务的数据交换格式需要统⼀ （接⼝的请求 / 响应格式等)

d. AI 应⽤接⼊⼯具的配置内容需要进⾏标准化定义

e. 所有⼯具服务必须提供标准化的接⼊⽅式，以⽀持通过标准化配置即可加载⼯具

f. 所有 AI 应⽤内部需实现标准化的⼯具加载调⽤逻辑，以⽀持通过标准化配置即可加载⼯具

## 系统架构设计

![](images/438b1bc890ba4caa316124603b81d7c4816342876648d0828b1958108f97e7ca.jpg)  
MCP 协议的雏型架构 ｜ by @bilibili 堂吉诃德拉曼查的英豪（Carbon Based）

## 3.3 MCP 协议是什么

## • 诞⽣：

2024 年 11 ⽉由 Anthropic（⼀家美国⼈⼯智能初创公司）提出，官⽅⽂档。

## • 定义：

MCP is an open protocol that standardizes how applications provide context to large language models (LLMs). Think of MCP like a USB-C port for AI applications. Just as USB-C provides a standardized way to connect your devices to various peripherals and accessories, MCP provides a standardized way to connect AI models to different data sources and tools. MCP enables you build agents and complex workflows on top of LLMs and connects your models with the world. [1]

MCP是⼀个开放协议，⽤于标准化应⽤程序向⼤语⾔模型 （LLM）提供上下⽂的⽅式。你可以把MCP 想象成 AI 应⽤的 USB-C 接⼝ 正如USB-C提供了⼀种将设备连接到各种外设和配件的标准化⽅式⼀样，MCP 提供了⼀种将 AI 模型连接到不同数据源和⼯具的标准化⽅式。借助 MCP，你可以在LLM之上构建智能体和复杂⼯作流，并将你的模型与外部世界相连接。

## • 如何理解：

◦ 应⽤程序：集成了LLM的具体应⽤。包括各家⼤模型的在线对话⽹站、集成了⼤模型的IDE（如 Claude desktop）、各种 Agent（⽐如 Cursor 就是⼀个 Agent）、以及其他接⼊了⼤模型的普通应⽤。

◦ 上下⽂：指的是模型在决策时可访问的所有信息，如当前⽤⼾输⼊、历史对话信息、外部⼯具（tool）信息、外部数据源（resource）信息、提⽰词（prompt）信息等等（这⾥重点只讲⼯具）。

![](images/92aeed809ed71f770f891e4c2bc566a2852bf49aa80af10af16553ca2e0fbb50.jpg)

和传统 API 的区别？

## 3.4 MCP 核⼼架构

MCP follows a client-server architecture where an MCP host — an AI application like Claude Code or Claude Desktop — establishes connections to one or more MCP servers. The MCP host accomplishes this by creating one MCP client for each MCP server. Each MCP client maintains a dedicated one-to-one connection with its corresponding MCP server.The key participants in the MCP architecture are:

• MCP Host: The AI application that coordinates and manages one or multiple MCP clients

• MCP Client: A component that maintains a connection to an MCP server and obtains context from an MCP server for the MCP host to use

• MCP Server: A program that provides context to MCP clients

For example: Visual Studio Code acts as an MCP host. When Visual Studio Code establishes a connection to an MCP server, such as the Sentry MCP server, the Visual Studio Code runtime instantiates an MCP client object that maintains the connection to the Sentry MCP server. When Visual Studio Code subsequently connects to another MCP server, such as the local filesystem server, the Visual Studio Code runtime instantiates an additional MCP client object to maintain this connection, hence maintaining a one-to-one relationship of MCP clients to MCP servers.Note that MCP server refers to the program that serves context data, regardless of where it runs. MCP servers can execute locally or remotely. For example, when Claude Desktop launches the filesystem server, the server runs locally on the same machine because it uses the STDIO transport. This is commonly referred to as a “local” MCP server. The officialSentry MCP server runs on the Sentry platform, and uses the Streamable HTTP transport. This is commonly referred to as a “remote” MCP server. [2]

MCP遵循客⼾端-服务器架构，其中 MCP Host Claude Code 或 Claude Desktop 等AI应⽤程序与⼀个或多个MCP Server 建⽴连接。MCP 主机通过为每个 MCP Server 创建⼀个 MCP Client 来实现这⼀⽬标。每个MCPClient都与相应的MCPServer保持专⽤的⼀对⼀连接。MCP架构的主要组成者是：

• MCP Host：协调和管理⼀个或多个 MCP Server 的⼈⼯智能应⽤程序

• MCPClient：⼀个组件，⽤于维护与MCP服务器的连接，并从MCP服务器获取上下⽂，供MCP主机使⽤

• MCP Server：⼀个为 MCP Client 提供上下⽂的程序例如：Visual Studio Code 充当 MCP 主机。当 Visual Studio Code 建⽴与MCP服务器（如SentryMCP服务器）的连接时，Visual Studio Code 运⾏时实例化了维护与Sentry MCP服务器连接的MCP客⼾端对象。当VisualStudioCode随后连接到另⼀个MCP服务器时，例如本地⽂件系统服务器，VisualStudioCode运⾏时实例化⼀个额外的MCP客⼾端对象来维护此连接，从⽽保持MCP客⼾端与MCP服务器的⼀对⼀关系。

![](images/7c2f1af1feaba6842809059d7ff0b6a44d9caeae06b02f272135c3aec8683c60.jpg)

## 3.5 MCP 的传输协议

MCP supports two transport mechanisms:

• Stdio transport: Uses standard input/output streams for direct process communication between local processes on the same machine, providing optimal performance with no network overhead.

• Streamable HTTP transport: Uses HTTP POST for client-to-server messages with optional Server-Sent Events for streaming capabilities. This transport enables remote server communication and supports standard HTTP authentication methods including bearer tokens, API keys, and custom headers. MCP recommends using OAuth to obtain authentication tokens.

The transport layer abstracts communication details from the protocol layer, enabling the same JSON-RPC 2.0 message format across all transport mechanisms. [3]

## Stdio 传输

Stdio 传输本质上是本地进程间通信（IPC）的⼀种形式，它最常⽤的底层机制就是管道（pipe）。

## 1. 什么是 stdio？

◦ stdio（standard I/O）是进程的标准输⼊/输出接⼝。每个进程启动时，操作系统会给它分配

三个⽂件描述符：

代码块

1 0 → stdin （标准输⼊，默认是键盘）  
2 1 → stdout （标准输出，默认是屏幕）  
3 2 → stderr （标准错误输出，默认是屏幕）

◦ 程序⾥的 printf 、 scanf 、 cin 、 cout 、 read 、 write 都是通过这些接⼝和外界交换数据的。

## 2. 什么是管道（pipe）？

◦ 管道是操作系统内核提供的⼀种进程间通信（IPC）机制，它允许⼀个进程的输出直接作为另⼀个进程的输⼊，实现数据在两个进程之间的流动。

## 3. 总结：什么是Stdio传输？

◦ 所谓Stdio传输，就是通过标准输⼊和标准输出这两个数据流来传输数据、通过管道来连接两个进程的标准输⼊/输出接⼝，使得⼀个进程的输出直接传给另⼀个进程输⼊，实现进程间数据传输（本质上是⼀个基于字节流的全双⼯通信通道）

◦ stdio是接⼝，管道是连接这接⼝的通道。

## • 举例：

命 令   
1 ps aux

## 没有⽤管道

[键盘] → shell → ps(stdin) → ps(stdout) → [屏幕]1

命令   
ps aux | grep python 1

## ⽤了管道（Stido 传输）

# ps 和 grep 在执⾏时都会各⾃成为⼀个独⽴的进程1

# ps aux 列出进程 → grep python 过滤后留下包含 "python" 的进程2

```markdown
# 管道 | 把 ps aux 的 stdout 作为 grep python 的 stdin 3
```

4 [键盘] → shell → ps(stdin) → ps(stdout) → grep(stdin) → grep(stdout) → 屏幕

![](images/5c5ba938584f2bca168094b34a1fa4e291f448b8e1334097daa0f44e9a1d5d8f.jpg)

为什么在这么多本地进程间通信的⽅式中选了 Stdio 传输？

## HTTP + SSE 传输（旧⽅案，2024.10）

客⼾端通过HTTPPOST向服务端发请求，服务端通过SSE通道返回响应结果。

• SSE（Server-SentEvents服务器发送事件），是⼀种服务器单向推送数据给客⼾端的技术，基于HTTP 协议。

• 基本原理

◦ 客⼾端先向服务端发起⼀个普通的HTTP请求。

◦ 服务端保持这个连接不断开，以 text/event-stream 作为响应类型，源源不断地往⾥写数据。

◦ 客⼾端收到数据后会触发相应的事件回调（⽐如浏览器前端实时更新界⾯）。

• 和普通HTTP的核⼼差异

◦ ⽀持服务端主动、流式地推送消息

## 为什么在这么多远程服务调⽤的协议中选了 HTTP + SSE？

## • 服务端推送的必要性：MCPServer中的⼯具发⽣了更新，需要主动向MCPClient推送通知

## Why Notifications Matter

This notification system is crucial for several reasons:

1. Dynamic Environments: Tools may come and go based on server state, external dependencies, or user permissions

2. Efficiency: Clients don’t need to poll for changes; they’re notified when updates occur

3. Consistency: Ensures clients always have accurate information about available server capabilities

4. Real-time Collaboration: Enables responsive AI applications that can adapt to changing contexts

This notification pattern extends beyond tools to other MCP primitives, enabling comprehensive real-time synchronization between clients and servers. [4]

## Streamable HTTP 传输（新⽅案，2025.03）

HTTP+SSE传输⽅案的升级版，⽬前正在逐步取代原有的HTTP+SSE传输⽅案

• StreamableHTTP并不是⼀个标准协议名，⽽是⼀个通⽤描述，指的是基于HTTP协议的“可流式传输”技术。它的核⼼思想是：在⼀个HTTP连接⾥，服务端可以持续不断地发送数据给客⼾端，客⼾端边接收边处理，类似“流”⼀样。与传统HTTP请求响应“⼀次性完成”不同，StreamableHTTP保持连接不关闭，数据分⽚持续传输。常⻅实现⽅式包括：

◦ HTTP/1.1 ⻓连接 + 分块传输编码（Chunked Transfer Encoding）

◦ HTTP/2 流式数据

◦ HTTP/3 QUIC 流式传输

## 为什么 HTTP + SSE 要升级成 Streamable HTTP ？

• 数据格式限制问题：SSE 的 Content-Type: text/event-stream 只⽀持⽂本格式；Streamable HTTP 的 Content-Type ⽀持任意格式，如 JSON、HTML、⼆进制等，更适合 AI场景（可能要传JSON+⾳频+图⽚）

• 跨平台兼容问题：SSE⽀持的客⼾端主要是浏览器端和少量语⾔库；⽽StreamableHTTP⽀持多种客⼾端。

• 性能问题：SSE 是基于 HTTP/1.1 ⻓连接，Streamable HTTP 可以基于 HTTP/2/3 ，⽀持多路复⽤和双向流。且HTTP/2/3的流控制和优先级机制使得⾼吞吐和低延迟成为可能；SSE消息只能⽂本格式，StreamableHTTP⽀持其他采⽤更紧凑的编码⽅式（⽐如⼆进制分包、压缩等）。

## 必须选⽤以上传输协议吗？

No，因为⽆论哪种传输⽅式，都只是把各种⼯具的不同接⼊⽅式统⼀起来，对外暴露⼀种协议的接⼝⽽已。

![](images/54b930d3f10a111f120c9d617ac9e0f0d97306067886aa90e1c0c10e78bc7887.jpg)  
MCP 协议的架构 ｜ by @bilibili 堂吉诃德拉曼查的英豪（Carbon Based）

![](images/ca0bfaa1776a533fb4cdeef24828788aa526656e02f67ec9ae8f5401810b0be4.jpg)

## 3.6回顾：技术⽅案最终是怎么实现的

1. ⼯具与AI 应⽤必须解耦合。

![](images/a519adf93fef90609f0d7efc1f3abf1916b18c41522a44b24e6128343b5be6fe.jpg)

✅ 客⼾端 - 服务端架构（MCP Host、MCP Client、MCP Server）

2. ⼯具与AI 应⽤之间的交互必须标准化。

a. AI应⽤和⼯具服务的通信协议需要统⼀ （本地进程间的通信协议/远程服务调⽤的协议）

![](images/b183fef08a4470a4a5d40c5ce7accbb19473996fc3784305a77088c10555dca0.jpg)

✅ 本地：Stdio 传输；远程：HTTP + SSE 或 Streamable HTTP

b. AI应⽤和⼯具服务的接⼝定义需要统 （需要提供哪些接⼝、接⼝需包含哪些参数）

![](images/5b4a81440677d77788552b5ab87dfe3d2e53e70f96a4bce1278a622470a8ed05.jpg)

✅ ⼯具服务需提供的接⼝：1. tools/list（⽤于返回⽅法列表） 2. tools/call（⽤于执⾏⽅法并返回结果。3. notifications/tools/list_changed（服务端主动推送，⽤于告知客⼾端⽅法更新）[5]

![](images/b0822987748cd47a453ec6c7a27948c734b025db20b163cdf00f68081033aac0.jpg)

✅ 接⼝参数定义，⻅⽂档。以tools/list为例：

## 3.1 Listing Tools

To discover available tools, clients send a tols/list request. This operation supports pagination.

## Request:

{   
"jsonrpc":"2.0",   
"method":"tools/list",   
"params":{   
"cursor":"optional-cursor-value"   
}   
}

## Response:

{   
"jsonrpc": "2.0",   
"name":"get_weather",   
"title":"Weather Information Provider",   
"description": "Get current weather information for a Location",   
"inputSchema":{   
"type":"object",   
"properties":{   
"location":{   
"type":"string",   
"description":"City name or zip code"   
}   
"required":["location"]   
}   
}   
"nextcursor":"next-page-cursor"   
}   
}

c. AI 应⽤和⼯具服务的数据交换格式需要统

## ✅ JSON-RPC 2.0 [6] [7]

JSON-RPC2.0是⼀种轻量级的远程过程调⽤（RPC）协议，基于JSON格式进⾏通信，主要特点是所有消息都是JSON格式，便于解析和跨语⾔使⽤。

## d. AI 应⽤接⼊⼯具的配置内容需要进⾏标准化定义

以接⼊⾼德地图 MCP Server 为例，参考⽂档：https://lbs.amap.com/api/mcp-server/gettingstarted（来⾃ModelScope 的 MCP Server 市场：https://modelscope.cn/mcp）

## 本地服务式接⼊（基于Stdio协议）

```json
Stdio⽅式接⼊配置
3
4 {
5 "mcpServers": {
6 "amap-maps": {
7 "command": "npx",
8 "args": ["-y", "@amap/amap-maps-mcp-server"],
9 "env": {
10 "AMAP_MAPS_API_KEY": "您在⾼德官⽹上申请的key"
11 }
12 }
13 }
14 }
```

![](images/7bc0b6f8524111700db2bdec3ec9aa5a021841d0874344a50e57b1f0d07dfe1d.jpg)

## 远程服务式接⼊（基于SSE 协议）

SSE ⽅式接⼊配置   
1 {   
2 "mcpServers": {   
3 "amap-maps-streamableHTTP": {   
4 "url": "https://mcp.amap.com/mcp?key=您在⾼德官⽹上申请的key"   
5 }   
6 }   
7 }

## e. 所有⼯具服务必须提供标准化的接⼊⽅式，以⽀持通过标准化配置即可加载⼯具

![](images/75b4ee310f46e9f71a05d7bfe5e874854db75f874c5ec281a61de790a7f53568.jpg)

✅ 遵循 MCP 协议开发 MCP Server，对外提供标准的接⼊⽅式。多语⾔ SDK 地址。

f. 所有AI应⽤内部需实现标准化的⼯具加载调⽤逻辑，以⽀持通过标准化配置即可加载⼯具

使⽤官⽅SDK，在AI应⽤后端项⽬中实例化MCPClient，调⽤SDK⽅法和Server交互。

## 3.7 MCP ⼯作流程

![](images/21da61d4f644d72052bb4863fc075f447da68218184e9315a0c99a0136635a25.jpg)

?? 值得⼀提的是，MCP 协议和 Function Calling 之间绝不是“技术递进”的关系。所谓“MCP 协议会取代 Function Calling”的说法，其实是⼀种不严谨的表达。

## 3.8 What's More?

• 看 MCP 协议官⽅⽂档：https://modelcontextprotocol.io/docs/getting-started/intro，思考为何这么设计；

• 通过官⽅ SDK，尝试编写⼀个 MCP Server 并启动服务；初始化⼀个后端应⽤，创建 MCP Client，完成 Client - Server 完整交互流程的开发。MCP SDK 地址：https://modelcontextprotocol.io/docs/sdk

• 找⼀个开源的⽀持了 MCP 协议的 Agent 框架，追溯其中涉及到 MCP Client 、MCP Server 逻辑的所 有 代 码 。