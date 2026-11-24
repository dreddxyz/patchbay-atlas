const pick=id=>document.getElementById(id);
async function start(){
 const response=await fetch('results.json');if(!response.ok)throw Error();
 const data=await response.json();
 const pct=v=>Math.round(v*100)+'%';
 pick('score').textContent='Вопросов: '+data.results.length+' · Верный первый результат: TF–IDF '+pct(data.metrics.tfidf.top1)+', MiniLM '+pct(data.metrics.dense.top1)+'.';
 for(const [i,r] of data.results.entries()){pick('question').add(new Option(r.query,i));}
 function show(){const r=data.results[Number(pick('question').value)];for(const [id,key] of [['lex','tfidf'],['dense','dense']]){const list=pick(id);list.replaceChildren();for(const hit of r[key]){const item=document.createElement('li');item.textContent=data.cards[hit.id].title+' · '+hit.score.toFixed(3);list.append(item);}}
 pick('expected').textContent='Эталонный источник: '+data.cards[r.expected].title;pick('answer').textContent=r.draft;pick('source').textContent=data.cards[r.dense[0].id].text;}
 pick('question').addEventListener('change',show);show();
}
start().catch(()=>pick('score').textContent='Не удалось загрузить результаты.');
