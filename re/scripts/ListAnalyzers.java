// Prints every auto-analyzer option for the current program as "name<TAB>value".
// Run as a headless -postScript after an -import with -noanalysis.
import java.util.Map;
import java.util.TreeMap;

import ghidra.app.util.headless.HeadlessScript;

public class ListAnalyzers extends HeadlessScript {
    @Override
    public void run() throws Exception {
        Map<String, String> options = new TreeMap<>(getCurrentAnalysisOptionsAndValues(currentProgram));
        for (Map.Entry<String, String> entry : options.entrySet()) {
            // Skip sub-options such as "Stack.Create Local Variables".
            if (!entry.getKey().contains(".")) {
                println("ANALYZER\t" + entry.getKey() + "\t" + entry.getValue());
            }
        }
        println("FUNCTIONS\t" + currentProgram.getFunctionManager().getFunctionCount());
        println("SYMBOLS\t" + currentProgram.getSymbolTable().getNumSymbols());
    }
}
