# NQ_multi_agent

# 目前的demo的使用指南

multi_agent.py里定义了三个agent: _extract_answer, _cut_answer, _refine_answer
目前三个agent的执行顺序是：
_extract_answer -> _cut_answer -> _refine_answer


# 如何测试当前代码
predictions = my_agnets.predict_batch(file, verbose=True)

输出：
Example 1/2
Question: can a ucc be filed on an individual
Initial answer: Yes, a UCC - 1 financing statement can be filed on an individual, as it is filed "in the state where the debtor resides."
Cut answer: Yes, a UCC - 1 financing statement can be filed on an individual.
Refined answer: Yes, a UCC can be filed on an individual.
Refined answer: Yes, a UCC can be filed on an individual.
- - - - - - - - - - 
5
gold_answer: []
final_answer: Yes, a UCC can be filed on an individual.

Example 2/2
Question: who is the guy who walked across the twin towers
Initial answer: Philippe Petit
Cut answer: Philippe Petit
Refined answer: Petit
Refined answer: Petit
- - - - - - - - - - 
5
gold_answer: ['Philippe Petit', 'Philippe Petit', 'Philippe Petit', 'Philippe Petit', 'Philippe Petit']
final_answer: Petit
