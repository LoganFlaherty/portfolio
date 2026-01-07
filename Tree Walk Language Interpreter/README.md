# Tree Walk Language Interpreter
In this project I successfully replicated a Java tree walk interpreter, following the book Crafting Interpreters, utilizing OpenJDK for a language called Lox. The interpreter features a 
scanner (lexer), parser, and AST-based interpreter to tokenize, parse, and execute code.This language up to this point is pretty simple only 
implementing dynamic variables, operands, printing, if statements, else statements, for loops, while loops, and the basics of scope in the form of 
nested blocks. To use the interpreter, download the 'Tree Walk Language Interpreter' folder on a windows 11 machine. Then open the project folder in 
a windows 11 terminal. Run the command .\make which will ensure OpenJDK21 is installed, all files are compiled, and execute the interpreter using the 
hello_world.lox file. If you want to pass your own file for testing, use the cmd java lox.Lox in the project file after the files have been compiled 
using the .\make cmd.
