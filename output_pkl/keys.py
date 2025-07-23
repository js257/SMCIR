from mmsdk import mmdatasdk

# 加载MOSI数据集中的视觉模态特征
visual_facet = 'CMU_MOSI_Visual_Facet_41.csd'
md = mmdatasdk.mmdataset({visual_facet: 'E:\cjs\mult\MMSA\src\MMSA\output_pkl\dataset\CMU_MOSI_Visual_Facet_41.csd'})

# 查看所有视频的ID
print(md.computational_sequences.keys())
