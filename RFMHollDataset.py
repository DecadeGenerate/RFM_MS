from data.Utils import *
import json
import mindspore
import x2ms_adapter
import x2ms_adapter.nn_functional


def get_selection_label(b, r, min_window_size=5, n_windows=4):
    # print(b.size())
    window_size = min_window_size
    bs = list()
    for i in range(n_windows):
        bs.append(x2ms_adapter.nn_functional.x2ms_pad(b.unfold(1, window_size, min_window_size), (0, min_window_size * n_windows - window_size)))
        window_size += min_window_size
    b_segments = x2ms_adapter.cat(bs, dim=1)

    b_list = x2ms_adapter.tensor_api.tolist(b_segments)
    r_list = x2ms_adapter.tensor_api.tolist(r)

    overlap = [[len(set(seg).intersection(r_list[i])) for seg in b_list[i]] for i in range(len(b_list))]

    p_s = x2ms_adapter.tensor_api.detach(x2ms_adapter.nn_functional.softmax(x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.x2ms_tensor(overlap)), dim=-1))
    return p_s


class RAMHollDataset:
    def __init__(self, files, vocab2id, min_window_size=5, num_windows=4, knowledge_len=300, context_len=65, n=1E10):
        super(RAMHollDataset, self).__init__()

        self.min_window_size = min_window_size
        self.num_windows = num_windows
        self.knowledge_len = knowledge_len
        self.context_len = context_len

        # 标量
        self.ids = list()
        self.contexts = list()
        self.queries = list()
        self.responses = list()
        self.unstructured_knowledges = list()
        self.dyn_vocab2ids = list()
        self.dyn_id2vocabs = list()
        self.example_id = list()

        # tensor
        self.id_arrays = list()
        self.context_arrays = list()
        self.query_arrays = list()
        self.response_arrays = list()
        self.dyn_response_arrays = list()
        self.unstructured_knowledge_arrays = list()

        self.ref_start_arrays = list()
        self.ref_end_arrays = list()

        self.dyn_map_arrays = list()
        self.vocab_map_arrays = list()
        self.vocab_overlap_arrays = list()

        self.selections = list()

        self.vocab2id = vocab2id
        self.files = files
        self.n = n

        self.load()

    def load(self):
        with codecs.open(self.files[0], encoding='utf-8') as f:
            data = json.load(f)
            for id in range(len(data)):
                sample = data[id]

                # 处理对话上下文
                contexts = sample['context']
                while len(contexts) < 2:
                    contexts = ['<nan>'] + contexts
                contexts += [sample['query']]
                contexts = x2ms_adapter.tensor_api.split(' '.join(contexts).lower(), ' ')
                self.contexts.append(contexts)
                contexts = contexts[-self.context_len:]
                contexts += [PAD_WORD] * (self.context_len - len(contexts))
                self.context_arrays.append(x2ms_adapter.tensor_api.long(x2ms_adapter.x2ms_tensor(
                    [self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in contexts],
                    requires_grad=False)))

                # 处理背景知识
                unstructured_knowledge_origin = x2ms_adapter.tensor_api.split(sample['unstructured_knowledge'].lower(), ' ')
                self.unstructured_knowledges.append(unstructured_knowledge_origin)
                unstructured_knowledge = unstructured_knowledge_origin[-self.knowledge_len:]
                while len(unstructured_knowledge) < self.knowledge_len:
                    unstructured_knowledge += [PAD_WORD]
                # unstructured_knowledge += [PAD_WORD] * (self.knowledge_len - len(unstructured_knowledge))
                b = x2ms_adapter.tensor_api.long(x2ms_adapter.x2ms_tensor([self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in
                                  unstructured_knowledge], requires_grad=False))
                if x2ms_adapter.tensor_api.x2ms_size(b)[0] > 256:
                    print(len(unstructured_knowledge))
                self.unstructured_knowledge_arrays.append(b)

                # 处理回复
                ress = sample['response']
                if isinstance(ress, list):
                    response = x2ms_adapter.tensor_api.split(ress[0].lower(), ' ')
                    self.responses.append([x2ms_adapter.tensor_api.split(r.lower(), ' ') for r in ress])
                else:
                    response = x2ms_adapter.tensor_api.split(ress.lower(), ' ')
                    self.responses.append([response])
                response = (response + [EOS_WORD])[:80]
                r = x2ms_adapter.tensor_api.long(x2ms_adapter.x2ms_tensor(
                    [self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in response],
                    requires_grad=False))
                self.response_arrays.append(r)

                self.selections.append(
                    get_selection_label(x2ms_adapter.tensor_api.unsqueeze(b, 0), x2ms_adapter.tensor_api.unsqueeze(r, 0), min_window_size=self.min_window_size,
                                        n_windows=self.num_windows))

                # from data.Utils import build_vocab
                dyn_vocab2id, dyn_id2vocab = build_vocab(unstructured_knowledge)  # 返回知识的vocab2id和id2vocab，特殊字符只有PAD
                self.dyn_vocab2ids.append(dyn_vocab2id)
                self.dyn_id2vocabs.append(dyn_id2vocab)

                self.dyn_response_arrays.append(
                    x2ms_adapter.tensor_api.long(x2ms_adapter.x2ms_tensor([dyn_vocab2id.get(w) if w in dyn_vocab2id else 0 for w in response],
                                 requires_grad=False)))
                self.dyn_map_arrays.append(
                    x2ms_adapter.x2ms_tensor([dyn_vocab2id.get(w) for w in unstructured_knowledge], requires_grad=False))

                vocab_map = []
                vocab_overlap = []
                for i in range(len(dyn_id2vocab)):
                    vocab_map.append(self.vocab2id.get(dyn_id2vocab[i], self.vocab2id[UNK_WORD]))  # 用背景知识用词表来表示
                    if dyn_id2vocab[i] in self.vocab2id:
                        vocab_overlap.append(0.)
                    else:
                        vocab_overlap.append(1.)
                self.vocab_map_arrays.append(x2ms_adapter.x2ms_tensor(vocab_map, requires_grad=False))  # 同上
                self.vocab_overlap_arrays.append(
                    x2ms_adapter.x2ms_tensor(vocab_overlap, requires_grad=False))  # 如果背景知识词在词表存在为0，否则为1

                if 'bg_ref_start' in sample:
                    print(sample['bg_ref_start'])
                    self.ref_start_arrays.append(x2ms_adapter.x2ms_tensor([sample['bg_ref_start']], requires_grad=False))
                    self.ref_end_arrays.append(x2ms_adapter.x2ms_tensor([sample['bg_ref_end'] - 1], requires_grad=False))
                else:
                    self.ref_start_arrays.append(x2ms_adapter.x2ms_tensor([-1], requires_grad=False))
                    self.ref_end_arrays.append(x2ms_adapter.x2ms_tensor([-1], requires_grad=False))

                e_id = sample['id']
                self.example_id.append(e_id)

                self.ids.append(id)
                self.id_arrays.append(x2ms_adapter.tensor_api.long(x2ms_adapter.x2ms_tensor([id])))

                if len(self.contexts) >= self.n:
                    break
        self.len = len(self.contexts)
        print('data size: ', self.len)

    def __getitem__(self, index):
        return [self.id_arrays[index], self.context_arrays[index], self.unstructured_knowledge_arrays[index],
                self.response_arrays[index], self.dyn_response_arrays[index], self.dyn_map_arrays[index],
                self.vocab_map_arrays[index], self.vocab_overlap_arrays[index],
                (self.ids[index], self.dyn_id2vocabs[index]), (self.ids[index], self.dyn_vocab2ids[index]),
                self.selections[index], self.ref_start_arrays[index], self.ref_end_arrays[index], self.example_id[index]]

    def __len__(self):
        return self.len

    def input(self, id):
        return self.contexts[id]

    def output(self, id):
        return self.responses[id]

    def background(self, id):
        return self.unstructured_knowledges[id]


def collate_fn(data):
    id_a, context_a, unstructured_knowledge_a, response_a, dyn_response_a, dyn_map, vocab_map, vocab_overlap, dyn_id2vocab, dyn_vocab2id, selection, ref_start, ref_end, example_id = zip(
        *data)

    return {'id': x2ms_adapter.cat(id_a),
            'context': pad_sequence(context_a, batch_first=True),
            'response': pad_sequence(response_a, batch_first=True),
            'unstructured_knowledge': pad_sequence(unstructured_knowledge_a, batch_first=True),
            'dyn_response': pad_sequence(dyn_response_a, batch_first=True),
            'dyn_map': pad_sequence(dyn_map, batch_first=True),
            'vocab_map': pad_sequence(vocab_map, batch_first=True),
            'vocab_overlap': pad_sequence(vocab_overlap, batch_first=True, padding_value=1.),
            'dyn_id2vocab': dict(dyn_id2vocab),
            'dyn_vocab2id': dict(dyn_vocab2id),
            'selection': x2ms_adapter.cat(selection),
            'ref_start': x2ms_adapter.cat(ref_start),
            'ref_end': x2ms_adapter.cat(ref_end),
            'example_id': example_id}
