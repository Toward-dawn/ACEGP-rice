# ACEGP GitHub代码包

此文件夹是拟上传GitHub的代码内容，当前仍为本地准备版，尚未上传。代码与环境文件共50份，全部与原v2公开包逐字节相同。

## 与数据包配合使用

将 `ACEGP_data_20261028_v2_split2.zip` 解压到本文件夹根目录，使 `01_code`、`02_data` 和 `04_documentation` 并列。不要套一层 `Zenodo_data` 文件夹。数据包与代码包没有同名冲突；`.gitignore` 排除了数据内容。

预测运行入口要求Linux/POSIX及指定依赖版本。可先运行英文README中的 `--check` 命令，它只核验文件和环境。完整运行命令见 `04_documentation/CODE_GUIDE.md`，其中训练和统计命令本次均未执行。

`FILE_MANIFEST.csv`是代码与数据合并后的运行清单，因此包含本代码包内暂不存在的 `02_data` 路径。这是配套数据清单，不是遗漏。代码包自身清单是 `CODE_FILE_MANIFEST.csv`，校验文件是 `SHA256SUMS_CODE.txt`。

validation随机对照的16份共同插补输入仍待补。已有100套面板和结果随数据包保留。两群体常规重复CV所用矩阵已收录。原代码中的development标识指primary population，内部区域ID和种子不改名。

原创代码已选择MIT，许可文本见LICENSE；版权人姓名仍明确标为待填。原创数据和结果采用CC BY 4.0，第三方资料保留原条款。作者、版权人及公开链接尚待确认，详见本次交付根目录下“仅供作者核对_不上传”。没有创建DOI或仓库地址，也未上传。
