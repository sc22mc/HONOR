def evaluate(self, eval_file, **judge_kwargs):
        infer_dataset = load(eval_file)

        rank, world_size = get_rank_and_world_size()
        if self.EVALUATE_METHOD == 'RULE':
            if rank == 0:
                res = self.evaluate_by_rule(infer_dataset, eval_file)

        elif self.EVALUATE_METHOD == 'LLM':
            res = self.evaluate_by_llm(infer_dataset, eval_file)

        else:
            raise Exception

        if rank != 0:
            return None
        
        self.calculate_results(eval_file, infer_dataset, res)
        
    def calculate_results(self, eval_file, infer_dataset, res):
        lines = [infer_dataset.iloc[i] for i in range(len(infer_dataset))]
        hit = self.hit_calculate(res, self.dataset_name)
        ret = dict()
        if 'split' in infer_dataset:
            splits = set(infer_dataset['split'])
            for sp in splits:
                sub = [r for l, r in zip(lines, res) if l['split'] == sp]
                # [np.mean(x['match']) >= full_score_weight for x in sub]
                hit = self.hit_calculate(sub, self.dataset_name)
                ret[sp] = np.mean(hit) * 100
            sub = [r for l, r in zip(lines, res)]
            hit = self.hit_calculate(sub, self.dataset_name)
            ret['Overall'] = np.mean(hit) * 100
        else:
            ret['Overall'] = np.mean(hit) * 100
            if 'category' in infer_dataset:
                cates = list(set(infer_dataset['category']))
                cates.sort()
                for c in cates:
                    sub = [r for l, r in zip(lines, res) if l['category'] == c]
                    # [np.mean(x['match']) >= full_score_weight for x in sub]
                    hit = self.hit_calculate(sub, self.dataset_name)
                    ret[c] = np.mean(hit) * 100
        ret = d2df(ret)
        ret.round(2)

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(ret, result_file)
        return ret
