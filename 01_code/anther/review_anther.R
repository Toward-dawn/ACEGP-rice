#!/usr/bin/env Rscript
# Refit only RNA-seq statistical models. Never load phenotype/prediction data.
args <- commandArgs(trailingOnly=TRUE)
suppressWarnings(Sys.setlocale('LC_CTYPE','English_United States.utf8'))
# Relative output paths avoid a Windows command-line filename encoding issue.
out <- args[1]
options(stringsAsFactors=FALSE, warn=1)
Sys.setenv(LANGUAGE="en")
suppressPackageStartupMessages({library(DESeq2);library(jsonlite);library(ggplot2);library(fgsea);library(limma);library(edgeR);library(BiocParallel)})
src <- fromJSON(file.path(out,"inputs/source_paths.json"))
savecsv <- function(x,p) write.csv(x,file.path(out,p),row.names=FALSE,na="")
savejson <- function(x,p) write_json(x,file.path(out,p),pretty=TRUE,auto_unbox=TRUE,na="null",digits=16)
set.seed(20261018)
packages <- c('DESeq2','edgeR','limma','fgsea','ggplot2','BiocParallel','jsonlite')
savecsv(data.frame(package=packages,version=sapply(packages,function(x)as.character(packageVersion(x)))),"versions/R_packages.csv")
writeLines(capture.output(sessionInfo()),file.path(out,"versions/sessionInfo.txt"))

raw <- read.table(src$counts,header=TRUE,row.names=1,check.names=FALSE,comment.char="#")
cts <- as.matrix(raw[,6:ncol(raw)])
colnames(cts) <- sub("_sorted.bam$","",basename(colnames(cts)))
meta <- read.csv(file.path(out,"inputs/sample_metadata.csv"),check.names=FALSE)
stopifnot(identical(colnames(cts),meta$sample),!anyNA(cts),all(cts>=0),all(cts==round(cts)))
coldata <- data.frame(row.names=meta$sample,cultivar=factor(meta$cultivar,levels=c('R4628','R996')),condition=factor(meta$condition,levels=c('control','heat')))
design <- model.matrix(~cultivar+condition+cultivar:condition,coldata)
stopifnot(qr(design)$rank==4,all(table(coldata$cultivar,coldata$condition)==3))
assign <- read.delim(src$featurecounts_summary,check.names=FALSE,row.names=1)
stopifnot(all(as.numeric(assign['Assigned',])==colSums(cts)))
qc <- cbind(meta,total_assigned_fragments=colSums(cts),featurecounts_assignment_rate=as.numeric(assign['Assigned',])/colSums(assign),detected_genes=colSums(cts>0))
for(i in seq_len(nrow(qc))){
 f <- paste0("fastp_",qc$sample[i],"_report")
 if(!is.null(src[[f]])){
  x <- fromJSON(src[[f]])
  qc$raw_reads[i]<-x$summary$before_filtering$total_reads
  qc$clean_reads[i]<-x$summary$after_filtering$total_reads
  qc$clean_Q30_rate[i]<-x$summary$after_filtering$q30_rate
  qc$clean_GC_fraction[i]<-x$summary$after_filtering$gc_content
 }
}
savecsv(qc,"tables/sample_QC.csv")
dds <- DESeqDataSetFromMatrix(countData=cts,colData=coldata,design=~cultivar+condition+cultivar:condition)
dds <- dds[rowSums(counts(dds))>=10,]
dds <- DESeq(dds,quiet=TRUE,parallel=FALSE)
stopifnot(identical(resultsNames(dds),c('Intercept','cultivar_R996_vs_R4628','condition_heat_vs_control','cultivarR996.conditionheat')))
res <- list(
 R996=results(dds,contrast=list(c('condition_heat_vs_control','cultivarR996.conditionheat')),alpha=0.05),
 R4628=results(dds,contrast=list(c('condition_heat_vs_control')),alpha=0.05),
 interaction=results(dds,name='cultivarR996.conditionheat',alpha=0.05))
norm <- counts(dds,normalized=TRUE)
savecsv(cbind(data.frame(gene_id=rownames(norm)),as.data.frame(norm)),"tables/normalized_counts.csv")
qc$size_factor<-sizeFactors(dds)
savecsv(qc,"tables/sample_QC.csv")
saveRDS(dds,file.path(out,'inputs/dds_recomputed.rds'))
allresults<-lapply(names(res),function(n){
 d<-as.data.frame(res[[n]]);d$gene_id<-rownames(d)
 d$standard_DEG<-!is.na(d$padj)&d$padj<0.05&abs(d$log2FoldChange)>1
 savecsv(d,paste0('tables/DESeq2_',n,'_all_genes.csv'));d
});names(allresults)<-names(res)
# Historical-file comparison omitted from the public workflow.
legacyset<-readLines(src$frozen_interaction_genes,warn=FALSE)
legacyset<-sub('^\ufeff','',legacyset);legacyset<-legacyset[nzchar(legacyset)]
nowset<-allresults$interaction$gene_id[allresults$interaction$standard_DEG]
genes<-read.csv(file.path(out,'inputs/frozen_51_anther_genes.csv'))
stopifnot(nrow(genes)==51)
master<-genes
for(n in names(res)){
 d<-allresults[[n]];ix<-match(master$gene_id,d$gene_id)
 for(k in c('log2FoldChange','lfcSE','stat','pvalue','padj','standard_DEG'))master[[paste0(n,'_',k)]]<-d[[k]][ix]
}
master$all_sample_baseMean<-res$interaction$baseMean[match(master$gene_id,rownames(res$interaction))]
for(g in c('R4628_control','R4628_heat','R996_control','R996_heat')){
 cols<-paste(coldata$cultivar,coldata$condition,sep='_')==g
 master[[paste0(g,'_normalized_mean')]]<-rowMeans(norm[master$gene_id,cols,drop=FALSE])
}
master$interaction_direction<-ifelse(master$interaction_log2FoldChange>0,'R996 heat response more positive','R4628 heat response more positive')
master$response_pattern<-ifelse(master$R996_log2FoldChange>0 & master$R4628_log2FoldChange<0,'R996 induction / R4628 repression',
 ifelse(master$R996_log2FoldChange<0 & master$R4628_log2FoldChange>0,'R996 repression / R4628 induction',
 ifelse(master$R996_log2FoldChange>0 & master$R4628_log2FoldChange>0,'both positive, magnitude differs','both negative, magnitude differs')))
master$R996_small_change_R4628_repressed<-abs(master$R996_log2FoldChange)<0.5 & master$R4628_log2FoldChange<(-1)&master$R4628_padj<0.05
master$R4628_small_change_R996_repressed<-abs(master$R4628_log2FoldChange)<0.5 & master$R996_log2FoldChange<(-1)&master$R996_padj<0.05
savecsv(master,'tables/51_genes_expression_recomputed.csv')
savejson(list(state='RNA_STATISTICAL_RECOMPUTATION_COMPLETE',raw_genes=nrow(cts),tested_genes=nrow(dds),original_global_interaction_genes=length(legacyset),recomputed_interaction_genes=length(nowset),frozen_global_list_confirmed=setequal(legacyset,nowset),frozen_51_confirmed=sum(master$interaction_standard_DEG),gene_id_compatibility_all51=all(genes$gene_id%in%rownames(dds)),gene_filter='rowSums raw counts>=10',batch_note='no documented batch variable; no invented covariate or sample exclusion',coefficient_identity_max_error=max(abs(res$R996$log2FoldChange-res$R4628$log2FoldChange-res$interaction$log2FoldChange),na.rm=TRUE),counting_note='legacy unstranded gene-body featureCounts -t gene -s 0; no re-counting or exon-level claim',mislabels=list(legacy_baseMean='R996_baseMean and R4628_baseMean in old master both represented the all-sample mean, not cultivar-specific means',legacy_additive='additive_heat_effect_all_genes.csv duplicated reference-cultivar heat contrast, not a separately fitted additive model',legacy_filter_log='printed baseMean>=10 although executed row sum>=10',interaction_sign='more positive relative response; not automatically favorable or tolerance-enhancing')), 'validation/RNA_RECOMPUTATION_ACCEPTANCE.json')

# QC plots; biological replicates shown individually. No sample exclusions.
vsd<-vst(dds,blind=TRUE)
v<-assay(vsd);top<-head(order(apply(v,1,var),decreasing=TRUE),500)
pc<-prcomp(t(v[top,,drop=FALSE]),center=TRUE,scale.=FALSE)
pct<-pc$sdev^2/sum(pc$sdev^2)*100
pdf<-cbind(meta,PC1=pc$x[,1],PC2=pc$x[,2])
savecsv(pdf,'tables/PCA_coordinates.csv')
theme_set(theme_bw(base_size=11)+theme(panel.grid.minor=element_blank()))
p<-ggplot(pdf,aes(PC1,PC2,color=cultivar,shape=condition))+geom_point(size=3)+geom_text(aes(label=paste0(substr(cultivar,2,8),'_',substr(condition,1,1),replicate)),vjust=-0.9,size=2.8,show.legend=FALSE)+scale_color_manual(values=c(R4628='#4F86A6',R996='#BF6B98'))+labs(x=sprintf('PC1 (%.1f%%)',pct[1]),y=sprintf('PC2 (%.1f%%)',pct[2]),title='Anther RNA-seq: 12 biological samples')
ggsave(file.path(out,'visualizations/anther_QC_PCA.png'),p,width=7,height=5,dpi=200)
ggsave(file.path(out,'visualizations/anther_QC_PCA.pdf'),p,width=7,height=5)
corv<-cor(v[top,,drop=FALSE]);savecsv(cbind(sample=rownames(corv),as.data.frame(corv)), 'tables/top500_sample_correlation.csv')
pv<-data.frame(sample=rep(colnames(corv),each=ncol(corv)),other=rep(rownames(corv),ncol(corv)),r=as.vector(corv))
p<-ggplot(pv,aes(sample,other,fill=r))+geom_tile()+scale_fill_gradient(low='#EAF0F3',high='#4F86A6')+coord_equal()+theme(axis.text.x=element_text(angle=65,hjust=1,size=8),axis.text.y=element_text(size=8))+labs(x=NULL,y=NULL,fill='Pearson r',title='VST expression correlation: 500 most variable genes')
ggsave(file.path(out,'visualizations/anther_sample_correlation.png'),p,width=8,height=7,dpi=180)

# Function sets are frozen before computing statistical scores.
ann<-read.csv(file.path(out,'inputs/functional_term_membership.csv'),check.names=FALSE)
termmeta<-ann[!duplicated(ann$term_id),c('term_id','term_name','domain','source')]
paths<-split(ann$gene_id,ann$term_id);paths<-lapply(paths,unique)
eligible<-rownames(dds)[is.finite(res$interaction$stat)]
global_sig<-nowset[nowset%in%eligible]
regionids<-read.csv(file.path(out,'inputs/all_development_region_genes.csv'))$gene_id
region_universe<-intersect(eligible,regionids)
savecsv(data.frame(gene_id=eligible,within_development_regions=eligible%in%regionids,global_interaction_DEG=eligible%in%global_sig), 'inputs/functional_tested_gene_universe.csv')
GSEA<-list();CAMERA<-list()
vo<-voom(DGEList(counts=counts(dds)),design=design,plot=FALSE)
for(n in names(res)){
 cat('Functional analyses:',n,'\n');flush.console()
 stats<-res[[n]]$stat;names(stats)<-rownames(res[[n]])
 stats<-stats[is.finite(stats)];stats<-stats[order(-stats,names(stats))]
 set.seed(20261018+match(n,names(res)))
 fg<-as.data.frame(fgseaMultilevel(pathways=paths,stats=stats,minSize=15,maxSize=500,eps=0,nPermSimple=1000,BPPARAM=SerialParam()))
 fg$leadingEdge<-vapply(fg$leadingEdge,paste,character(1),collapse=';');fg$contrast<-n
 fg$within_call_padj<-fg$padj;fg$padj<-NULL
 fg<-merge(fg,termmeta,by.x='pathway',by.y='term_id',all.x=TRUE,sort=FALSE)
 GSEA[[n]]<-fg
 idx<-ids2indices(paths,rownames(vo))
 idx<-idx[lengths(idx)>=15 & lengths(idx)<=500]
 contrast<-switch(n,R996=c(0,0,1,1),R4628=c(0,0,1,0),interaction=c(0,0,0,1))
 ca<-camera(vo,index=idx,design=design,contrast=contrast,inter.gene.cor=NA,allow.neg.cor=FALSE,sort=FALSE)
 ca$term_id<-rownames(ca);ca$contrast<-n;ca$within_call_FDR<-ca$FDR;ca$FDR<-NULL
 ca<-merge(ca,termmeta,by='term_id',all.x=TRUE,sort=FALSE)
 CAMERA[[n]]<-ca
}
fg<-do.call(rbind,GSEA);fg$joint_BH_FDR<-p.adjust(fg$pval,method='BH');fg<-fg[order(fg$joint_BH_FDR,fg$pval),]
ca<-do.call(rbind,CAMERA);ca$joint_BH_FDR<-p.adjust(ca$PValue,method='BH');ca<-ca[order(ca$joint_BH_FDR,ca$PValue),]
savecsv(fg,'functional/GSEA_all_three_contrasts_joint_BH.csv')
savecsv(ca,'functional/CAMERA_all_three_contrasts_joint_BH.csv')
oratargets<-list(global_interaction=global_sig,regional_51=intersect(genes$gene_id,region_universe))
universe<-list(global_interaction=eligible,regional_51=region_universe)
OR<-list()
for(n in names(oratargets)){
 univ<-universe[[n]];sel<-intersect(oratargets[[n]],univ)
 pp<-lapply(paths,intersect,univ);pp<-pp[lengths(pp)>=3 & lengths(pp)<=500]
 tab<-do.call(rbind,lapply(names(pp),function(id){
  hit<-intersect(sel,pp[[id]]);K<-length(pp[[id]]);k<-length(hit);N<-length(univ);m<-length(sel)
  data.frame(target=n,term_id=id,universe_n=N,selected_n=m,term_universe_n=K,overlap_n=k,fold_enrichment=(k/m)/(K/N),pvalue=phyper(k-1,K,N-K,m,lower.tail=FALSE),overlap_gene_ids=paste(sort(hit),collapse=';'))
 }))
 OR[[n]]<-tab
}
ora<-do.call(rbind,OR);ora$joint_BH_FDR<-p.adjust(ora$pvalue,method='BH');ora<-merge(ora,termmeta,by='term_id',all.x=TRUE,sort=FALSE);ora<-ora[order(ora$joint_BH_FDR,ora$pvalue),]
savecsv(ora,'functional/ORA_global_and_region_joint_BH.csv')
fsummary<-rbind(
 do.call(rbind,lapply(split(fg,fg$contrast),function(z)data.frame(method='ranked_GSEA',comparison=z$contrast[1],tests=nrow(z),FDR_lt_005=sum(z$joint_BH_FDR<0.05,na.rm=TRUE)))),
 do.call(rbind,lapply(split(ca,ca$contrast),function(z)data.frame(method='CAMERA_correlation_sensitivity',comparison=z$contrast[1],tests=nrow(z),FDR_lt_005=sum(z$joint_BH_FDR<0.05,na.rm=TRUE)))),
 do.call(rbind,lapply(split(ora,ora$target),function(z)data.frame(method='ORA',comparison=z$target[1],tests=nrow(z),FDR_lt_005=sum(z$joint_BH_FDR<0.05,na.rm=TRUE)))))
savecsv(fsummary,'functional/functional_analysis_summary.csv')
savejson(list(state='FUNCTIONAL_ANALYSES_COMPLETE',GSEA_tests=nrow(fg),CAMERA_tests=nrow(ca),ORA_tests=nrow(ora),eligible_RNA_genes=length(eligible),within_region_RNA_universe=length(region_universe),regional_selected=length(oratargets$regional_51),annotation_mapped_GO=length(intersect(eligible,unique(ann$gene_id[ann$source=='GO']))),annotation_mapped_KEGG=length(intersect(eligible,unique(ann$gene_id[ann$source=='KEGG']))),seed=20261018,prediction_models_run=0), 'functional/FUNCTIONAL_ACCEPTANCE.json')
cat('ANALYSIS_COMPLETE: RNA contrasts, QC and functional analyses\n')
