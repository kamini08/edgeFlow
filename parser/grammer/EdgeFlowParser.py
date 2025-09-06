# Generated from grammer/EdgeFlow.g4 by ANTLR 4.13.1
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,19,81,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,1,0,4,0,26,8,0,11,
        0,12,0,27,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,41,8,1,
        1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1,4,1,4,1,5,1,5,1,5,1,5,1,6,1,6,
        1,6,1,6,1,7,1,7,1,7,1,7,1,8,1,8,1,8,1,8,1,9,1,9,1,9,1,9,1,10,1,10,
        1,10,1,10,1,11,1,11,1,11,1,11,1,11,0,0,12,0,2,4,6,8,10,12,14,16,
        18,20,22,0,1,1,0,11,13,77,0,25,1,0,0,0,2,40,1,0,0,0,4,42,1,0,0,0,
        6,46,1,0,0,0,8,50,1,0,0,0,10,52,1,0,0,0,12,56,1,0,0,0,14,60,1,0,
        0,0,16,64,1,0,0,0,18,68,1,0,0,0,20,72,1,0,0,0,22,76,1,0,0,0,24,26,
        3,2,1,0,25,24,1,0,0,0,26,27,1,0,0,0,27,25,1,0,0,0,27,28,1,0,0,0,
        28,29,1,0,0,0,29,30,5,0,0,1,30,1,1,0,0,0,31,41,3,4,2,0,32,41,3,6,
        3,0,33,41,3,10,5,0,34,41,3,12,6,0,35,41,3,14,7,0,36,41,3,16,8,0,
        37,41,3,18,9,0,38,41,3,20,10,0,39,41,3,22,11,0,40,31,1,0,0,0,40,
        32,1,0,0,0,40,33,1,0,0,0,40,34,1,0,0,0,40,35,1,0,0,0,40,36,1,0,0,
        0,40,37,1,0,0,0,40,38,1,0,0,0,40,39,1,0,0,0,41,3,1,0,0,0,42,43,5,
        2,0,0,43,44,5,1,0,0,44,45,5,16,0,0,45,5,1,0,0,0,46,47,5,3,0,0,47,
        48,5,1,0,0,48,49,3,8,4,0,49,7,1,0,0,0,50,51,7,0,0,0,51,9,1,0,0,0,
        52,53,5,4,0,0,53,54,5,1,0,0,54,55,5,15,0,0,55,11,1,0,0,0,56,57,5,
        5,0,0,57,58,5,1,0,0,58,59,5,16,0,0,59,13,1,0,0,0,60,61,5,6,0,0,61,
        62,5,1,0,0,62,63,5,15,0,0,63,15,1,0,0,0,64,65,5,7,0,0,65,66,5,1,
        0,0,66,67,5,17,0,0,67,17,1,0,0,0,68,69,5,8,0,0,69,70,5,1,0,0,70,
        71,5,15,0,0,71,19,1,0,0,0,72,73,5,9,0,0,73,74,5,1,0,0,74,75,5,17,
        0,0,75,21,1,0,0,0,76,77,5,10,0,0,77,78,5,1,0,0,78,79,5,14,0,0,79,
        23,1,0,0,0,2,27,40
    ]

class EdgeFlowParser ( Parser ):

    grammarFileName = "EdgeFlow.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'='", "'model'", "'quantize'", "'target_device'", 
                     "'deploy_path'", "'input_stream'", "'buffer_size'", 
                     "'optimize_for'", "'memory_limit'", "'enable_fusion'", 
                     "'int8'", "'float16'", "'none'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "MODEL", "QUANTIZE", "TARGET_DEVICE", 
                      "DEPLOY_PATH", "INPUT_STREAM", "BUFFER_SIZE", "OPTIMIZE_FOR", 
                      "MEMORY_LIMIT", "FUSION", "INT8", "FLOAT16", "NONE", 
                      "BOOL", "IDENTIFIER", "STRING", "INTEGER", "COMMENT", 
                      "WS" ]

    RULE_program = 0
    RULE_statement = 1
    RULE_modelStmt = 2
    RULE_quantizeStmt = 3
    RULE_quantType = 4
    RULE_targetDeviceStmt = 5
    RULE_deployPathStmt = 6
    RULE_inputStreamStmt = 7
    RULE_bufferSizeStmt = 8
    RULE_optimizeForStmt = 9
    RULE_memoryLimitStmt = 10
    RULE_fusionStmt = 11

    ruleNames =  [ "program", "statement", "modelStmt", "quantizeStmt", 
                   "quantType", "targetDeviceStmt", "deployPathStmt", "inputStreamStmt", 
                   "bufferSizeStmt", "optimizeForStmt", "memoryLimitStmt", 
                   "fusionStmt" ]

    EOF = Token.EOF
    T__0=1
    MODEL=2
    QUANTIZE=3
    TARGET_DEVICE=4
    DEPLOY_PATH=5
    INPUT_STREAM=6
    BUFFER_SIZE=7
    OPTIMIZE_FOR=8
    MEMORY_LIMIT=9
    FUSION=10
    INT8=11
    FLOAT16=12
    NONE=13
    BOOL=14
    IDENTIFIER=15
    STRING=16
    INTEGER=17
    COMMENT=18
    WS=19

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class ProgramContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EOF(self):
            return self.getToken(EdgeFlowParser.EOF, 0)

        def statement(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(EdgeFlowParser.StatementContext)
            else:
                return self.getTypedRuleContext(EdgeFlowParser.StatementContext,i)


        def getRuleIndex(self):
            return EdgeFlowParser.RULE_program

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterProgram" ):
                listener.enterProgram(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitProgram" ):
                listener.exitProgram(self)




    def program(self):

        localctx = EdgeFlowParser.ProgramContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_program)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 25 
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 24
                self.statement()
                self.state = 27 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not ((((_la) & ~0x3f) == 0 and ((1 << _la) & 2044) != 0)):
                    break

            self.state = 29
            self.match(EdgeFlowParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class StatementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def modelStmt(self):
            return self.getTypedRuleContext(EdgeFlowParser.ModelStmtContext,0)


        def quantizeStmt(self):
            return self.getTypedRuleContext(EdgeFlowParser.QuantizeStmtContext,0)


        def targetDeviceStmt(self):
            return self.getTypedRuleContext(EdgeFlowParser.TargetDeviceStmtContext,0)


        def deployPathStmt(self):
            return self.getTypedRuleContext(EdgeFlowParser.DeployPathStmtContext,0)


        def inputStreamStmt(self):
            return self.getTypedRuleContext(EdgeFlowParser.InputStreamStmtContext,0)


        def bufferSizeStmt(self):
            return self.getTypedRuleContext(EdgeFlowParser.BufferSizeStmtContext,0)


        def optimizeForStmt(self):
            return self.getTypedRuleContext(EdgeFlowParser.OptimizeForStmtContext,0)


        def memoryLimitStmt(self):
            return self.getTypedRuleContext(EdgeFlowParser.MemoryLimitStmtContext,0)


        def fusionStmt(self):
            return self.getTypedRuleContext(EdgeFlowParser.FusionStmtContext,0)


        def getRuleIndex(self):
            return EdgeFlowParser.RULE_statement

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterStatement" ):
                listener.enterStatement(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitStatement" ):
                listener.exitStatement(self)




    def statement(self):

        localctx = EdgeFlowParser.StatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_statement)
        try:
            self.state = 40
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [2]:
                self.enterOuterAlt(localctx, 1)
                self.state = 31
                self.modelStmt()
                pass
            elif token in [3]:
                self.enterOuterAlt(localctx, 2)
                self.state = 32
                self.quantizeStmt()
                pass
            elif token in [4]:
                self.enterOuterAlt(localctx, 3)
                self.state = 33
                self.targetDeviceStmt()
                pass
            elif token in [5]:
                self.enterOuterAlt(localctx, 4)
                self.state = 34
                self.deployPathStmt()
                pass
            elif token in [6]:
                self.enterOuterAlt(localctx, 5)
                self.state = 35
                self.inputStreamStmt()
                pass
            elif token in [7]:
                self.enterOuterAlt(localctx, 6)
                self.state = 36
                self.bufferSizeStmt()
                pass
            elif token in [8]:
                self.enterOuterAlt(localctx, 7)
                self.state = 37
                self.optimizeForStmt()
                pass
            elif token in [9]:
                self.enterOuterAlt(localctx, 8)
                self.state = 38
                self.memoryLimitStmt()
                pass
            elif token in [10]:
                self.enterOuterAlt(localctx, 9)
                self.state = 39
                self.fusionStmt()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ModelStmtContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def MODEL(self):
            return self.getToken(EdgeFlowParser.MODEL, 0)

        def STRING(self):
            return self.getToken(EdgeFlowParser.STRING, 0)

        def getRuleIndex(self):
            return EdgeFlowParser.RULE_modelStmt

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterModelStmt" ):
                listener.enterModelStmt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitModelStmt" ):
                listener.exitModelStmt(self)




    def modelStmt(self):

        localctx = EdgeFlowParser.ModelStmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_modelStmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 42
            self.match(EdgeFlowParser.MODEL)
            self.state = 43
            self.match(EdgeFlowParser.T__0)
            self.state = 44
            self.match(EdgeFlowParser.STRING)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class QuantizeStmtContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def QUANTIZE(self):
            return self.getToken(EdgeFlowParser.QUANTIZE, 0)

        def quantType(self):
            return self.getTypedRuleContext(EdgeFlowParser.QuantTypeContext,0)


        def getRuleIndex(self):
            return EdgeFlowParser.RULE_quantizeStmt

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterQuantizeStmt" ):
                listener.enterQuantizeStmt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitQuantizeStmt" ):
                listener.exitQuantizeStmt(self)




    def quantizeStmt(self):

        localctx = EdgeFlowParser.QuantizeStmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_quantizeStmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 46
            self.match(EdgeFlowParser.QUANTIZE)
            self.state = 47
            self.match(EdgeFlowParser.T__0)
            self.state = 48
            self.quantType()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class QuantTypeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def INT8(self):
            return self.getToken(EdgeFlowParser.INT8, 0)

        def FLOAT16(self):
            return self.getToken(EdgeFlowParser.FLOAT16, 0)

        def NONE(self):
            return self.getToken(EdgeFlowParser.NONE, 0)

        def getRuleIndex(self):
            return EdgeFlowParser.RULE_quantType

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterQuantType" ):
                listener.enterQuantType(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitQuantType" ):
                listener.exitQuantType(self)




    def quantType(self):

        localctx = EdgeFlowParser.QuantTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_quantType)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 50
            _la = self._input.LA(1)
            if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 14336) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TargetDeviceStmtContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def TARGET_DEVICE(self):
            return self.getToken(EdgeFlowParser.TARGET_DEVICE, 0)

        def IDENTIFIER(self):
            return self.getToken(EdgeFlowParser.IDENTIFIER, 0)

        def getRuleIndex(self):
            return EdgeFlowParser.RULE_targetDeviceStmt

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTargetDeviceStmt" ):
                listener.enterTargetDeviceStmt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTargetDeviceStmt" ):
                listener.exitTargetDeviceStmt(self)




    def targetDeviceStmt(self):

        localctx = EdgeFlowParser.TargetDeviceStmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_targetDeviceStmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 52
            self.match(EdgeFlowParser.TARGET_DEVICE)
            self.state = 53
            self.match(EdgeFlowParser.T__0)
            self.state = 54
            self.match(EdgeFlowParser.IDENTIFIER)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class DeployPathStmtContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def DEPLOY_PATH(self):
            return self.getToken(EdgeFlowParser.DEPLOY_PATH, 0)

        def STRING(self):
            return self.getToken(EdgeFlowParser.STRING, 0)

        def getRuleIndex(self):
            return EdgeFlowParser.RULE_deployPathStmt

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterDeployPathStmt" ):
                listener.enterDeployPathStmt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitDeployPathStmt" ):
                listener.exitDeployPathStmt(self)




    def deployPathStmt(self):

        localctx = EdgeFlowParser.DeployPathStmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_deployPathStmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 56
            self.match(EdgeFlowParser.DEPLOY_PATH)
            self.state = 57
            self.match(EdgeFlowParser.T__0)
            self.state = 58
            self.match(EdgeFlowParser.STRING)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class InputStreamStmtContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def INPUT_STREAM(self):
            return self.getToken(EdgeFlowParser.INPUT_STREAM, 0)

        def IDENTIFIER(self):
            return self.getToken(EdgeFlowParser.IDENTIFIER, 0)

        def getRuleIndex(self):
            return EdgeFlowParser.RULE_inputStreamStmt

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterInputStreamStmt" ):
                listener.enterInputStreamStmt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitInputStreamStmt" ):
                listener.exitInputStreamStmt(self)




    def inputStreamStmt(self):

        localctx = EdgeFlowParser.InputStreamStmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_inputStreamStmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 60
            self.match(EdgeFlowParser.INPUT_STREAM)
            self.state = 61
            self.match(EdgeFlowParser.T__0)
            self.state = 62
            self.match(EdgeFlowParser.IDENTIFIER)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class BufferSizeStmtContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def BUFFER_SIZE(self):
            return self.getToken(EdgeFlowParser.BUFFER_SIZE, 0)

        def INTEGER(self):
            return self.getToken(EdgeFlowParser.INTEGER, 0)

        def getRuleIndex(self):
            return EdgeFlowParser.RULE_bufferSizeStmt

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBufferSizeStmt" ):
                listener.enterBufferSizeStmt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBufferSizeStmt" ):
                listener.exitBufferSizeStmt(self)




    def bufferSizeStmt(self):

        localctx = EdgeFlowParser.BufferSizeStmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_bufferSizeStmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 64
            self.match(EdgeFlowParser.BUFFER_SIZE)
            self.state = 65
            self.match(EdgeFlowParser.T__0)
            self.state = 66
            self.match(EdgeFlowParser.INTEGER)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class OptimizeForStmtContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def OPTIMIZE_FOR(self):
            return self.getToken(EdgeFlowParser.OPTIMIZE_FOR, 0)

        def IDENTIFIER(self):
            return self.getToken(EdgeFlowParser.IDENTIFIER, 0)

        def getRuleIndex(self):
            return EdgeFlowParser.RULE_optimizeForStmt

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterOptimizeForStmt" ):
                listener.enterOptimizeForStmt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitOptimizeForStmt" ):
                listener.exitOptimizeForStmt(self)




    def optimizeForStmt(self):

        localctx = EdgeFlowParser.OptimizeForStmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_optimizeForStmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 68
            self.match(EdgeFlowParser.OPTIMIZE_FOR)
            self.state = 69
            self.match(EdgeFlowParser.T__0)
            self.state = 70
            self.match(EdgeFlowParser.IDENTIFIER)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MemoryLimitStmtContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def MEMORY_LIMIT(self):
            return self.getToken(EdgeFlowParser.MEMORY_LIMIT, 0)

        def INTEGER(self):
            return self.getToken(EdgeFlowParser.INTEGER, 0)

        def getRuleIndex(self):
            return EdgeFlowParser.RULE_memoryLimitStmt

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMemoryLimitStmt" ):
                listener.enterMemoryLimitStmt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMemoryLimitStmt" ):
                listener.exitMemoryLimitStmt(self)




    def memoryLimitStmt(self):

        localctx = EdgeFlowParser.MemoryLimitStmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_memoryLimitStmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 72
            self.match(EdgeFlowParser.MEMORY_LIMIT)
            self.state = 73
            self.match(EdgeFlowParser.T__0)
            self.state = 74
            self.match(EdgeFlowParser.INTEGER)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class FusionStmtContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def FUSION(self):
            return self.getToken(EdgeFlowParser.FUSION, 0)

        def BOOL(self):
            return self.getToken(EdgeFlowParser.BOOL, 0)

        def getRuleIndex(self):
            return EdgeFlowParser.RULE_fusionStmt

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFusionStmt" ):
                listener.enterFusionStmt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFusionStmt" ):
                listener.exitFusionStmt(self)




    def fusionStmt(self):

        localctx = EdgeFlowParser.FusionStmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_fusionStmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 76
            self.match(EdgeFlowParser.FUSION)
            self.state = 77
            self.match(EdgeFlowParser.T__0)
            self.state = 78
            self.match(EdgeFlowParser.BOOL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





