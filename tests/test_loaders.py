from core.loaders.markdown import MarkdownLoader
loader = MarkdownLoader()
print(loader.supports("test.md")) # 应该输出 True