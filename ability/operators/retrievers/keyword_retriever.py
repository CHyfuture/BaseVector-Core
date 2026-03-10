"""Keyword retriever."""
import re
from typing import Any, Dict, List, Optional

from ability.config import get_settings
from ability.operators.retrievers.base_retriever import (
    BaseRetriever,
    RetrievalResult,
    metadata_from_result,
    resolve_output_fields,
)
from ability.storage.milvus_client import milvus_client
from ability.utils.logger import logger

settings = get_settings()


class KeywordRetriever(BaseRetriever):
    """Keyword retriever based on token matching and BM25-like scoring."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.min_match_count = self.get_config("min_match_count", 1)
        self.tf_normalization_factor = self.get_config("tf_normalization_factor", 100.0)
        self.candidate_multiplier = self.get_config("candidate_multiplier", 10)
        self.max_tokens = self.get_config("max_tokens", 32)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer."""
        chinese_pattern = r"[\u4e00-\u9fff]+"
        english_pattern = r"[a-zA-Z0-9]+"

        tokens = []
        for match in re.finditer(chinese_pattern, text):
            seq = match.group()
            if len(seq) <= 2:
                tokens.append(seq)
            else:
                tokens.extend(seq[i : i + 2] for i in range(len(seq) - 1))
        for match in re.finditer(english_pattern, text):
            tokens.append(match.group().lower())

        deduped = []
        seen = set()
        for token in tokens:
            if token and token not in seen:
                seen.add(token)
                deduped.append(token)

        if self.max_tokens and len(deduped) > self.max_tokens:
            deduped = deduped[: self.max_tokens]

        return deduped

    def _calculate_keyword_score(self, query_tokens: List[str], content: str) -> float:
        """Compute keyword match score."""
        content_tokens = self._tokenize(content)
        content_lower = content.lower()

        score = 0.0
        matched_count = 0

        for token in query_tokens:
            token_lower = token.lower()
            # 闂佽崵濮崇欢銈囨閺囥垺鍋╃紒顐ょ殹ken闂備線娼荤拹鐔煎礉瀹€鍕埞閻庣數纭堕崑鎾广亹閹哄棗浜炬繛鎴炵矊楠炩偓闂備礁鎲￠崹鐓庘枍閿濆宓侀柛銉墯閸庡秹鏌涢弴銊ョ伇闁哄棎鍎甸弻?
            count = content_lower.count(token_lower)
            if count > 0:
                matched_count += 1
                # 濠电偠鎻紞鈧繛澶嬫礋瀵偊濡堕崶鈺€绗夐梺閫炲苯澧寸€规洩绻濆鎾偐瀹曞洦娈搁梺鑽ゅТ濞层倗绮旇ぐ鎺嬧偓鍌炲醇閺囩偟楠囬梺鍛婂姦閸犳牕鈻?
                score += count * (1.0 / (1.0 + len(content_tokens) / self.tf_normalization_factor))

        # 濠电姷顣介埀顒€鍟块埀顒€缍婇幃妯诲緞閹邦剛顦遍悷婊呭鐢帞鏁幎鑺ョ厽闁靛鍎遍鈺呮煕濞呰娲﹂悡銉︺亜閺冨洦纭舵俊灞傚€曢…鎸庯紣濠靛懐鐩庨梺娲荤厛閸ㄥ爼寮鍛殕闁告劦浜為澶愭⒑?
        if matched_count < self.min_match_count:
            return 0.0

        # 闁荤喐绮庢晶妤冩暜婵犲嫮鍗氶柟缁㈠枛缁€宀勬煛瀹擃喖鍟伴埀顒€顭烽弻?
        if len(query_tokens) > 0:
            score = score / len(query_tokens)

        return score

    def _retrieve(
        query: str,
        top_k: int,
        tenant_id: Optional[str],
        **kwargs,
    ) -> List[RetrievalResult]:
        """Keyword retrieval implementation."""
        # 1. 缂備胶铏庨崣搴ㄥ窗閺嶎厽鍋╁Δ锝呭暞閳锋牠鏌涢埄鍐炬畷闁稿鍊濋弻娑橆潩椤掑倐銈囨偖閵娾晜鐓ユ繛鎴炵懁娴溿垽鏌ｉ妸褍鏋涢柟顔荤矙婵℃悂鏁傞悾灞藉箥濠电偟顥愰崑鎰叏閹绢喖鐭楅柛鈩兠杈ㄤ繆椤栨碍鎯堥悽顖樺劦閺屻劌鈽夊Ο鍨伃闂佽鍨欢姘暦濮橆兘鏋庨柟閭﹀墰濡棗鈹戦埥鍡楃仚闁告挻绻傞…?+ tenant_id
        collection_name = kwargs.get("collection_name")
        if not collection_name:
            if tenant_id:
                collection_name = settings.MILVUS_COLLECTION_TEMPLATE.format(tenant_id=tenant_id)
            else:
                collection_name = settings.MILVUS_COLLECTION_TEMPLATE.format(
                    tenant_id=settings.DEFAULT_TENANT_ID
                )

        # 2. 闂佽绨肩徊濠氾綖婢舵劖鍋傞柨娑樺鐎氭岸鏌曡箛銉ф偧缂佺姴顭烽幃璺衡槈濡灝顏梺鎼炲妼濞寸兘骞?
        query_tokens = self._tokenize(query)
        if not query_tokens:
            logger.warning("Query has no valid tokens after tokenization")
            return []

        # 3. 闂備礁鎼鍛偓姘煎墰缁辨捇骞樼€涙ü姘﹂梺鎼炲劘閸斿秵鎱ㄩ姀銈嗗仩婵炴垶鐗曟慨鈧梺瑙勬た娴滄粓顢?
        expr = None

        # 3.1 闂備胶顭堢换鎰版偋婵犲嫭鏆滈柟缁㈠枛缁€鍌炴煏婢跺牆鈧鎮烽幇顓犵闁瑰墽绮▍婊呯磼閸楃偟鍩ｇ€规洘顨婃俊鐤槼闁?milvus_expr闂備焦瀵х粙鎴︽偋閸涱垱顐介柕澶涚細缁?'chunk_type == "child"'闂?
        user_expr = kwargs.get("milvus_expr")
        if user_expr:
            expr = str(user_expr)

        # 闂備礁鎼鍛偓姘煎墰缁辨捇骞樼拠鑼槯闂佽鍎兼慨銈夊汲濞戙垺鍋ｉ柛銉戝憛銉х磼鐠囨彃鏆欐い顓滃姂婵″爼宕煎☉鎺戜壕闁告劏鏅滅紞鍥煙椤栧棗鎳愰、鍛存⒑閹稿海鈽夐柣妤€鎳忕换娑㈠炊椤掍礁浠洪梺鎰佸亝閳瑰儓E闂備胶鎳撻悘婵堢矓瀹曞洨绀婇柡鍐ㄥ€荤粻鏃堟煥閺冨洤浜圭紒鈧?
        keyword_filters = []
        for token in query_tokens:
            # 闂佸搫顦遍崕鎰板礂濞戞氨涓嶉柣鏂垮悑閸嬪鎮峰▎蹇擃伀闁哄棙娲熼幃妤€鈽夊▎妯荤暭濡?
            escaped_token = token.replace('"', '\\"').replace("'", "\\'")
            # 濠电偠鎻紞鈧繛澶嬫礋瀵偊藟閻攤E闂佸搫顦弲婊呯矙閺嶎厹鈧線骞嬪顏嗗枛閺佸秹宕熼鍕喘閺屾盯寮借缁夋椽鏌?
            keyword_filters.append(f'content like "%{escaped_token}%"')

        if keyword_filters:
            keyword_expr = " || ".join(f"({f})" for f in keyword_filters)
            if expr:
                # 闂備礁鎲￠懝楣冨嫉椤掆偓椤啴宕掑鑲╁弳闂傚嫬娲畷妯肩磼濡皷鏋栨繝鐢靛Т閸熶即宕?milvus_expr 濠电偞鍨堕幐鍛婎殽閹间礁鐭楃憸鏃堝蓟閵娾晩鏁婇悹鎭掑妽琚氶梻浣告啞閻楁洜绱炴繝鍥у偍闁靛牆顦痪褔鏌嶉妷銉ョ骇妞?
                expr = f"({expr}) && ({keyword_expr})"
            else:
                expr = keyword_expr

        # 4. 闂佽绻愮换鎴犳崲閸℃稒鍎婃い鏍ㄧ⊕婵挳鎮归幁鎺戝闁哄棗绮癷lvus闂備礁鎼悮顐﹀磿閹绢噮鏁嬫俊銈呮噺閺咁剟鏌涢锝囩婵炲眰鍊濋弻锛勨偓锝庡亞閻鏌℃笟鍥ф珝妤犵偞鍔栫粙澶愬椤ゆ厫闂備胶鎳撻悘婵堢矓瀹曞洨绀婇柡鍐ㄥ€荤粻鏃堟煥閺冨洤浜圭紒鈧?
        try:
            output_fields = resolve_output_fields(
                collection_name,
                kwargs.get("output_fields"),
            )
            collection = milvus_client.get_collection(collection_name)
            collection.load()

            # 濠电偠鎻紞鈧繛澶嬫礋瀵偊藝濮婄尃ry闂備礁鎼崐浠嬶綖婢跺本鍏滈柛顐ｆ礀閽冪喖鏌曟径妯煎帥闁搞倕瀚伴弻鐔哄枈濡桨澹曢梻浣告惈閻楀棝藝椤栨粈鐒婃い鏇楀亾闁哄苯锕ら濂稿炊閵娧勬闂備礁鎼崐绋棵洪敃鍌毼ラ柛宀€鍋為弲顒勬煕椤愶絿鐭岄柣鐔村灲濮婂宕卞Δ渚囨闂?schema 濠电偞鍨堕幐鎾磻閹剧粯鐓犻柣銈庡灡瑜把呯磼鏉堛劎绠炵€规洘绻堟俊鎼佹晜閻熼澹?output_fields闂?
            query_results = collection.query(
                expr=expr,
                output_fields=["id", *output_fields],
                limit=top_k * self.candidate_multiplier,  # 闂備礁鍚嬮崕鎶藉床閼艰翰浜归柛銉墮閸楁娊鏌℃径瀣劸婵☆垰顑夐弻娑欑節閸愵亝鍣伴梺閫炲苯澧い锕佷含閸掓帡濡搁埡浣稿殤濠电姴艌閸嬫挸霉濠婂嫬绗氱紒妤冨枛楠炴劖鎯旈敍鍕噰闂?
            )

            if not query_results:
                return []

            # 5. 闂佽閰ｅ褍锕㈡潏鈺佸灊闁靛ň鏅涢崙鐘崇節婵犲倹顥犵紒鐘差煼閹泛鈽夊Ο鍨伃闂佸憡顏搁崶銊у弳濡炪倖妫佸▔鏇炍ｉ妶澶嬪仯闁搞儜鍐句患闂佹悶鍔岄柊锝呯暦椤忓棔娌柟顖嗗懐鎳囬梺?
            scored_results = []
            for result in query_results:
                content = result.get("content", "")
                if not content:
                    continue

                score = self._calculate_keyword_score(query_tokens, content)

                if score > 0:
                    doc_id = result.get("doc_id") or result.get("document_id", 0)
                    scored_results.append(
                        {
                            "chunk_id": result.get("id"),
                            "document_id": doc_id,
                            "content": content,
                            "score": score,
                            "metadata": metadata_from_result(result),
                        }
                    )

            # 闂備礁婀遍…鍫ニ囬鐐插瀭閹兼番鍔岄弸渚€鏌ｅΔ鈧悧鍡欑箔閹捐绠归柡澶嬪灩缁犱即鏌ら懡銈呮瀾婵炵厧顭峰顒勫箰鎼达綆妲瑃op_k
            scored_results.sort(key=lambda x: x["score"], reverse=True)
            top_results = scored_results[:top_k]

            # 6. 闂備礁鎼鍛偓姘煎墰缁辨捇骞樺鍕洴閸┾偓妞ゆ帒鍊归～鏇㈡煏韫囧鐏繛澶堝灲閺?
            results = []
            for result_data in top_results:
                retrieval_result = RetrievalResult(
                    chunk_id=result_data["chunk_id"],
                    document_id=result_data["document_id"],
                    content=result_data["content"],
                    score=result_data["score"],
                    metadata=result_data["metadata"],
                )
                results.append(retrieval_result)

            return results

        except Exception as e:
            logger.warning(f"Keyword retrieval failed for collection {collection_name}: {str(e)}")
            # 濠电姷顣介埀顒€鍟块埀顒€缍婇幃妯荤箙婵犵拰E闂備胶鎳撻悘婵堢矓瀹曞洨绀婇柡鍐ㄥ€荤粻鏃堟煥閺冨洦纭堕柣鐔哥箞閺岋繝鍩€椤掑嫷鏁嶆繛鎴烆焽濡茬兘姊洪幐搴ｂ槈闁活厺鑳堕幑銏ゅ箣閿曗偓閻愬﹪鏌ｉ幇顕呮毌闁轰礁瀚伴弻娑㈠箳閹垮啯鐣介梺?
            return []
