// EdgeAILang.g4 - ANTLR Grammar for Edge AI DSL

grammar EdgeAILang;

// Parser rules

program
    : statement+ EOF
    ;

statement
    : modelStmt
    | quantizeStmt
    | targetDeviceStmt
    | deployPathStmt
    | inputStreamStmt
    | bufferSizeStmt
    ;

modelStmt
    : MODEL ':' STRING
    ;

quantizeStmt
    : QUANTIZE ':' quantType
    ;

quantType
    : INT8
    | FLOAT16
    | NONE
    ;

targetDeviceStmt
    : TARGET_DEVICE ':' IDENTIFIER
    ;

deployPathStmt
    : DEPLOY_PATH ':' STRING
    ;

inputStreamStmt
    : INPUT_STREAM ':' IDENTIFIER
    ;

bufferSizeStmt
    : BUFFER_SIZE ':' INTEGER
    ;

// Lexer rules

MODEL           : 'model';
QUANTIZE        : 'quantize';
TARGET_DEVICE   : 'target_device';
DEPLOY_PATH     : 'deploy_path';
INPUT_STREAM    : 'input_stream';
BUFFER_SIZE     : 'buffer_size';

INT8            : 'int8';
FLOAT16         : 'float16';
NONE            : 'none';

IDENTIFIER      : [a-zA-Z_] [a-zA-Z_0-9]* ;
STRING          : '"' (~["\r\n])* '"' ;
INTEGER         : [0-9]+ ;


WS              : [ \t\r\n]+ -> skip ;
