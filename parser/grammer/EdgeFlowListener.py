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



del EdgeFlowParser