grammar EdgeFlow;

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
    : MODEL_PATH '=' STRING
    ;

quantizeStmt
    : QUANTIZE '=' quantType
    ;

quantType
    : INT8
    | FLOAT16
    | NONE
    ;

targetDeviceStmt
    : TARGET_DEVICE '=' IDENTIFIER
    ;

deployPathStmt
    : DEPLOY_PATH '=' STRING
    ;

inputStreamStmt
    : INPUT_STREAM '=' IDENTIFIER
    ;

bufferSizeStmt
    : BUFFER_SIZE '=' INTEGER
    ;

// Lexer rules

MODEL_PATH      : 'model_path';
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
BOOL            : 'true' | 'false';

COMMENT         : '//' .*? '\n' -> skip;
WS              : [ \t\r\n]+ -> skip ;
