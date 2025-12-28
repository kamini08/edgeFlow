# Generated from grammer/EdgeFlow.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .EdgeFlowParser import EdgeFlowParser
else:
    from EdgeFlowParser import EdgeFlowParser

# This class defines a complete listener for a parse tree produced by EdgeFlowParser.
class EdgeFlowListener(ParseTreeListener):

    # Enter a parse tree produced by EdgeFlowParser#program.
    def enterProgram(self, ctx:EdgeFlowParser.ProgramContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#program.
    def exitProgram(self, ctx:EdgeFlowParser.ProgramContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#statement.
    def enterStatement(self, ctx:EdgeFlowParser.StatementContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#statement.
    def exitStatement(self, ctx:EdgeFlowParser.StatementContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#modelStmt.
    def enterModelStmt(self, ctx:EdgeFlowParser.ModelStmtContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#modelStmt.
    def exitModelStmt(self, ctx:EdgeFlowParser.ModelStmtContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#quantizeStmt.
    def enterQuantizeStmt(self, ctx:EdgeFlowParser.QuantizeStmtContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#quantizeStmt.
    def exitQuantizeStmt(self, ctx:EdgeFlowParser.QuantizeStmtContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#quantType.
    def enterQuantType(self, ctx:EdgeFlowParser.QuantTypeContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#quantType.
    def exitQuantType(self, ctx:EdgeFlowParser.QuantTypeContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#targetDeviceStmt.
    def enterTargetDeviceStmt(self, ctx:EdgeFlowParser.TargetDeviceStmtContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#targetDeviceStmt.
    def exitTargetDeviceStmt(self, ctx:EdgeFlowParser.TargetDeviceStmtContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#deployPathStmt.
    def enterDeployPathStmt(self, ctx:EdgeFlowParser.DeployPathStmtContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#deployPathStmt.
    def exitDeployPathStmt(self, ctx:EdgeFlowParser.DeployPathStmtContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#inputStreamStmt.
    def enterInputStreamStmt(self, ctx:EdgeFlowParser.InputStreamStmtContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#inputStreamStmt.
    def exitInputStreamStmt(self, ctx:EdgeFlowParser.InputStreamStmtContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#bufferSizeStmt.
    def enterBufferSizeStmt(self, ctx:EdgeFlowParser.BufferSizeStmtContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#bufferSizeStmt.
    def exitBufferSizeStmt(self, ctx:EdgeFlowParser.BufferSizeStmtContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#optimizeForStmt.
    def enterOptimizeForStmt(self, ctx:EdgeFlowParser.OptimizeForStmtContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#optimizeForStmt.
    def exitOptimizeForStmt(self, ctx:EdgeFlowParser.OptimizeForStmtContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#memoryLimitStmt.
    def enterMemoryLimitStmt(self, ctx:EdgeFlowParser.MemoryLimitStmtContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#memoryLimitStmt.
    def exitMemoryLimitStmt(self, ctx:EdgeFlowParser.MemoryLimitStmtContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#fusionStmt.
    def enterFusionStmt(self, ctx:EdgeFlowParser.FusionStmtContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#fusionStmt.
    def exitFusionStmt(self, ctx:EdgeFlowParser.FusionStmtContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#frameworkStmt.
    def enterFrameworkStmt(self, ctx:EdgeFlowParser.FrameworkStmtContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#frameworkStmt.
    def exitFrameworkStmt(self, ctx:EdgeFlowParser.FrameworkStmtContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#hybridOptimizationStmt.
    def enterHybridOptimizationStmt(self, ctx:EdgeFlowParser.HybridOptimizationStmtContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#hybridOptimizationStmt.
    def exitHybridOptimizationStmt(self, ctx:EdgeFlowParser.HybridOptimizationStmtContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#pytorchQuantizeStmt.
    def enterPytorchQuantizeStmt(self, ctx:EdgeFlowParser.PytorchQuantizeStmtContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#pytorchQuantizeStmt.
    def exitPytorchQuantizeStmt(self, ctx:EdgeFlowParser.PytorchQuantizeStmtContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#pytorchQuantType.
    def enterPytorchQuantType(self, ctx:EdgeFlowParser.PytorchQuantTypeContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#pytorchQuantType.
    def exitPytorchQuantType(self, ctx:EdgeFlowParser.PytorchQuantTypeContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#fineTuningStmt.
    def enterFineTuningStmt(self, ctx:EdgeFlowParser.FineTuningStmtContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#fineTuningStmt.
    def exitFineTuningStmt(self, ctx:EdgeFlowParser.FineTuningStmtContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#pruningStmt.
    def enterPruningStmt(self, ctx:EdgeFlowParser.PruningStmtContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#pruningStmt.
    def exitPruningStmt(self, ctx:EdgeFlowParser.PruningStmtContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#pruningConfig.
    def enterPruningConfig(self, ctx:EdgeFlowParser.PruningConfigContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#pruningConfig.
    def exitPruningConfig(self, ctx:EdgeFlowParser.PruningConfigContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#schedulingStmt.
    def enterSchedulingStmt(self, ctx:EdgeFlowParser.SchedulingStmtContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#schedulingStmt.
    def exitSchedulingStmt(self, ctx:EdgeFlowParser.SchedulingStmtContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#schedulingConfig.
    def enterSchedulingConfig(self, ctx:EdgeFlowParser.SchedulingConfigContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#schedulingConfig.
    def exitSchedulingConfig(self, ctx:EdgeFlowParser.SchedulingConfigContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#schedulingParamList.
    def enterSchedulingParamList(self, ctx:EdgeFlowParser.SchedulingParamListContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#schedulingParamList.
    def exitSchedulingParamList(self, ctx:EdgeFlowParser.SchedulingParamListContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#schedulingParam.
    def enterSchedulingParam(self, ctx:EdgeFlowParser.SchedulingParamContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#schedulingParam.
    def exitSchedulingParam(self, ctx:EdgeFlowParser.SchedulingParamContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#resourceConstraintsStmt.
    def enterResourceConstraintsStmt(self, ctx:EdgeFlowParser.ResourceConstraintsStmtContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#resourceConstraintsStmt.
    def exitResourceConstraintsStmt(self, ctx:EdgeFlowParser.ResourceConstraintsStmtContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#constraintList.
    def enterConstraintList(self, ctx:EdgeFlowParser.ConstraintListContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#constraintList.
    def exitConstraintList(self, ctx:EdgeFlowParser.ConstraintListContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#constraint.
    def enterConstraint(self, ctx:EdgeFlowParser.ConstraintContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#constraint.
    def exitConstraint(self, ctx:EdgeFlowParser.ConstraintContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#deploymentConfigStmt.
    def enterDeploymentConfigStmt(self, ctx:EdgeFlowParser.DeploymentConfigStmtContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#deploymentConfigStmt.
    def exitDeploymentConfigStmt(self, ctx:EdgeFlowParser.DeploymentConfigStmtContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#deploymentParamList.
    def enterDeploymentParamList(self, ctx:EdgeFlowParser.DeploymentParamListContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#deploymentParamList.
    def exitDeploymentParamList(self, ctx:EdgeFlowParser.DeploymentParamListContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#deploymentParam.
    def enterDeploymentParam(self, ctx:EdgeFlowParser.DeploymentParamContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#deploymentParam.
    def exitDeploymentParam(self, ctx:EdgeFlowParser.DeploymentParamContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#pipelineConfigStmt.
    def enterPipelineConfigStmt(self, ctx:EdgeFlowParser.PipelineConfigStmtContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#pipelineConfigStmt.
    def exitPipelineConfigStmt(self, ctx:EdgeFlowParser.PipelineConfigStmtContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#pipelineParamList.
    def enterPipelineParamList(self, ctx:EdgeFlowParser.PipelineParamListContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#pipelineParamList.
    def exitPipelineParamList(self, ctx:EdgeFlowParser.PipelineParamListContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#pipelineParam.
    def enterPipelineParam(self, ctx:EdgeFlowParser.PipelineParamContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#pipelineParam.
    def exitPipelineParam(self, ctx:EdgeFlowParser.PipelineParamContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#pipelineStmt.
    def enterPipelineStmt(self, ctx:EdgeFlowParser.PipelineStmtContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#pipelineStmt.
    def exitPipelineStmt(self, ctx:EdgeFlowParser.PipelineStmtContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#pipelineAttrList.
    def enterPipelineAttrList(self, ctx:EdgeFlowParser.PipelineAttrListContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#pipelineAttrList.
    def exitPipelineAttrList(self, ctx:EdgeFlowParser.PipelineAttrListContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#pipelineAttr.
    def enterPipelineAttr(self, ctx:EdgeFlowParser.PipelineAttrContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#pipelineAttr.
    def exitPipelineAttr(self, ctx:EdgeFlowParser.PipelineAttrContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#pipelineBody.
    def enterPipelineBody(self, ctx:EdgeFlowParser.PipelineBodyContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#pipelineBody.
    def exitPipelineBody(self, ctx:EdgeFlowParser.PipelineBodyContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#declInput.
    def enterDeclInput(self, ctx:EdgeFlowParser.DeclInputContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#declInput.
    def exitDeclInput(self, ctx:EdgeFlowParser.DeclInputContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#declOutput.
    def enterDeclOutput(self, ctx:EdgeFlowParser.DeclOutputContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#declOutput.
    def exitDeclOutput(self, ctx:EdgeFlowParser.DeclOutputContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#tensorType.
    def enterTensorType(self, ctx:EdgeFlowParser.TensorTypeContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#tensorType.
    def exitTensorType(self, ctx:EdgeFlowParser.TensorTypeContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#dimList.
    def enterDimList(self, ctx:EdgeFlowParser.DimListContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#dimList.
    def exitDimList(self, ctx:EdgeFlowParser.DimListContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#dim.
    def enterDim(self, ctx:EdgeFlowParser.DimContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#dim.
    def exitDim(self, ctx:EdgeFlowParser.DimContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#layerDecl.
    def enterLayerDecl(self, ctx:EdgeFlowParser.LayerDeclContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#layerDecl.
    def exitLayerDecl(self, ctx:EdgeFlowParser.LayerDeclContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#layerType.
    def enterLayerType(self, ctx:EdgeFlowParser.LayerTypeContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#layerType.
    def exitLayerType(self, ctx:EdgeFlowParser.LayerTypeContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#argList.
    def enterArgList(self, ctx:EdgeFlowParser.ArgListContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#argList.
    def exitArgList(self, ctx:EdgeFlowParser.ArgListContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#arg.
    def enterArg(self, ctx:EdgeFlowParser.ArgContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#arg.
    def exitArg(self, ctx:EdgeFlowParser.ArgContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#argValue.
    def enterArgValue(self, ctx:EdgeFlowParser.ArgValueContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#argValue.
    def exitArgValue(self, ctx:EdgeFlowParser.ArgValueContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#constrainedNumber.
    def enterConstrainedNumber(self, ctx:EdgeFlowParser.ConstrainedNumberContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#constrainedNumber.
    def exitConstrainedNumber(self, ctx:EdgeFlowParser.ConstrainedNumberContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#positiveInt.
    def enterPositiveInt(self, ctx:EdgeFlowParser.PositiveIntContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#positiveInt.
    def exitPositiveInt(self, ctx:EdgeFlowParser.PositiveIntContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#kernelSize.
    def enterKernelSize(self, ctx:EdgeFlowParser.KernelSizeContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#kernelSize.
    def exitKernelSize(self, ctx:EdgeFlowParser.KernelSizeContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#strideValue.
    def enterStrideValue(self, ctx:EdgeFlowParser.StrideValueContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#strideValue.
    def exitStrideValue(self, ctx:EdgeFlowParser.StrideValueContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#poolSize.
    def enterPoolSize(self, ctx:EdgeFlowParser.PoolSizeContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#poolSize.
    def exitPoolSize(self, ctx:EdgeFlowParser.PoolSizeContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#dropoutRate.
    def enterDropoutRate(self, ctx:EdgeFlowParser.DropoutRateContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#dropoutRate.
    def exitDropoutRate(self, ctx:EdgeFlowParser.DropoutRateContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#activationType.
    def enterActivationType(self, ctx:EdgeFlowParser.ActivationTypeContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#activationType.
    def exitActivationType(self, ctx:EdgeFlowParser.ActivationTypeContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#paddingType.
    def enterPaddingType(self, ctx:EdgeFlowParser.PaddingTypeContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#paddingType.
    def exitPaddingType(self, ctx:EdgeFlowParser.PaddingTypeContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#valueList.
    def enterValueList(self, ctx:EdgeFlowParser.ValueListContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#valueList.
    def exitValueList(self, ctx:EdgeFlowParser.ValueListContext):
        pass


    # Enter a parse tree produced by EdgeFlowParser#connectionStmt.
    def enterConnectionStmt(self, ctx:EdgeFlowParser.ConnectionStmtContext):
        pass

    # Exit a parse tree produced by EdgeFlowParser#connectionStmt.
    def exitConnectionStmt(self, ctx:EdgeFlowParser.ConnectionStmtContext):
        pass



del EdgeFlowParser