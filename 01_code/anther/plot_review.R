args<-commandArgs(trailingOnly=TRUE)
suppressWarnings(Sys.setlocale('LC_CTYPE','English_United States.utf8'))
out<-args[1]
suppressPackageStartupMessages(library(ggplot2))
theme_set(theme_bw(base_size=10)+theme(panel.grid=element_blank()))
master<-read.csv(file.path(out,'tables/51_genes_complete_evidence_and_SNP_table.csv'),check.names=FALSE)
master$label<-ifelse(is.na(master$gene_symbol)|master$gene_symbol=='',master$gene_id,paste0(master$gene_id,'  ',master$gene_symbol))
lev<-rev(master$label)
hm<-do.call(rbind,lapply(c('R996','R4628','interaction'),function(k){
 data.frame(gene_id=master$gene_id,label=factor(master$label,levels=lev),region=master$qtl_name,comparison=k,log2FC=master[[paste0(k,'_log2FoldChange')]],FDR=master[[paste0(k,'_padj')]])
}))
hm$comparison<-factor(hm$comparison,levels=c('R996','R4628','interaction'),labels=c('R996\nheat/control','R4628\nheat/control','Interaction\nR996 minus R4628'))
bound<-ceiling(max(abs(hm$log2FC)))
p<-ggplot(hm,aes(comparison,label,fill=log2FC))+geom_tile(color='white',linewidth=.35)+
 geom_point(data=hm[!is.na(hm$FDR)&hm$FDR<.05,],shape=16,size=.8,color='#27343B')+
 scale_fill_gradient2(low='#4F86A6',mid='white',high='#BF6B98',midpoint=0,limits=c(-bound,bound))+
 scale_x_discrete(position='top')+
 labs(x=NULL,y=NULL,fill=expression(log[2]*FC),title='51 anther interaction genes within fixed QTL regions',subtitle='Heat response differs by genotype; sign does not establish a beneficial effect',caption='Solid dots: BH FDR < 0.05 for the displayed contrast. No predictive model was fitted.')+
 theme(axis.text.y=element_text(size=7.5),axis.text.x=element_text(size=9),legend.position='right',plot.caption=element_text(hjust=0,size=8))
ggsave(file.path(out,'visualizations/anther_51_genes_heat_response.png'),p,width=8.8,height=13,dpi=200)
ggsave(file.path(out,'visualizations/anther_51_genes_heat_response.pdf'),p,width=8.8,height=13)

fg<-read.csv(file.path(out,'functional/GSEA_all_three_contrasts_joint_BH.csv'))
ids<-c('osa03010','GO:0006364','osa00190','osa01200','osa00500','osa04141','osa00480')
titles<-c('Ribosome','rRNA processing','Oxidative phosphorylation','Carbon metabolism','Starch and sucrose metabolism','ER protein processing','Glutathione metabolism')
d<-fg[fg$pathway%in%ids,]
d$term<-factor(titles[match(d$pathway,ids)],levels=rev(titles))
d$contrast<-factor(d$contrast,levels=c('R996','R4628','interaction'),labels=c('R996 heat/control','R4628 heat/control','Interaction'))
d$label<-paste0(sprintf('%.2f',d$NES),ifelse(d$joint_BH_FDR<.05,' *',''))
b<-ceiling(max(abs(d$NES)))
p<-ggplot(d,aes(contrast,term,fill=NES))+geom_tile(color='white',linewidth=1)+geom_text(aes(label=label),size=3.2)+
 scale_fill_gradient2(low='#4F86A6',mid='white',high='#BF6B98',midpoint=0,limits=c(-b,b))+
 labs(x=NULL,y=NULL,title='Whole-anther ranked gene-set analysis',subtitle='Values: normalized enrichment score; * joint BH FDR < 0.05',caption='CAMERA correlation sensitivity: no gene set at joint FDR < 0.05. These are pathway clues, not validated mechanisms.')+
 theme(axis.text.x=element_text(angle=15,hjust=.6),legend.position='right',plot.caption=element_text(size=8,hjust=0))
ggsave(file.path(out,'visualizations/anther_pathway_evidence_overview.png'),p,width=9.6,height=5.5,dpi=200)
ggsave(file.path(out,'visualizations/anther_pathway_evidence_overview.pdf'),p,width=9.6,height=5.5)
cat('PLOTS_COMPLETE\n')
