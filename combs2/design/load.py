


class Load:

    def __init__(self):
        pass





class Load:
    """Doesn't yet deal with terminal residues (although phi/psi does)"""

    def __init__(self, **kwargs):
        self.name = kwargs.get('name')
        self.path = kwargs.get('path', './')  # path to sig reps
        # self.designable = list()  # list of tuples (segment, chain, residue number)
        self.sequence_csts = kwargs.get('sequence_csts')  # keys1 are tuples (seq, ch, #), keys2 are label,
        # vals are allowed residue names (three letter code).
        self.dataframe = pd.DataFrame()
        self.dataframe_grouped = None
        self._rot = defaultdict(dict)
        self._mobile_com = defaultdict(dict)
        self._target_com = defaultdict(dict)
        self._sig_reps = defaultdict(dict)
        self._ideal_ala_df = defaultdict(dict)
        self._nonclashing = list()
        self.filetype = kwargs.get('filetype', '.feather')
        self.remove_from_df = kwargs.get('remove_from_df')  # e.g. {1: {'chain': 'Y', 'name': 'CB', 'resname': 'ASN'},
        #       2: {'chain': 'Y', 'name': 'CG', 'resname': 'GLN'}}

    @staticmethod
    def _get_targ_coords(template, label, seg, chain, resnum):
        sel_str = 'segment ' + seg + ' chain ' + chain + ' resnum ' + str(resnum) + ' name '
        cs = []
        for n in rel_coords_dict[label]:
            try:
                cs.append(template.pdb.select(sel_str + n).getCoords()[0])
            except AttributeError:
                try:
                    cs = []
                    for n in ['N', '1H', 'CA']:
                        cs.append(template.pdb.select(sel_str + n).getCoords()[0])
                    return np.stack(cs)
                except AttributeError:
                    try:
                        cs = []
                        for n in ['N', 'H1', 'CA']:
                            cs.append(template.pdb.select(sel_str + n).getCoords()[0])
                        return np.stack(cs)
                    except AttributeError:
                        sel_str = 'chain ' + chain + ' resnum ' + str(resnum) + ' name '
                        cs = []
                        for n in rel_coords_dict[label]:
                            try:
                                cs.append(template.pdb.select(sel_str + n).getCoords()[0])
                            except AttributeError:
                                cs = []
                                for n in ['N', '1H', 'CA']:
                                    cs.append(template.pdb.select(sel_str + n).getCoords()[0])
                                return np.stack(cs)
                        return np.stack(cs)
        return np.stack(cs)

        # return np.array([template.pdb.select(sel_str + n).getCoords()[0]
        #                 for n in rel_coords_dict[label]])

    @staticmethod
    def _get_mob_coords(df, label):
        # cs = []
        # for n in rel_coords_dict[label]:
        #     try:
        #         cs.append(df[df['name'] == n][['c_x', 'c_y', 'c_z']].values.flatten())
        #     except AttributeError:
        #         cs = []
        #         for n in ['N', 'H1', 'CA']:
        #             cs.append(df[df['name'] == n][['c_x', 'c_y', 'c_z']].values.flatten())
        # return np.stack(cs)
        return np.stack(df[df['name'] == n][['c_x', 'c_y', 'c_z']].values.flatten()
                        for n in rel_coords_dict[label])

    def set_rot_trans(self, template):
        for seg, chain, resnum in self.sequence_csts.keys():
            for label, df in df_ideal_ala.items():
                mob_coords = self._get_mob_coords(df, label)
                targ_coords = self._get_targ_coords(template, label, seg, chain, resnum)
                R, m_com, t_com = get_rot_trans(mob_coords, targ_coords)
                self._rot[label][(seg, chain, resnum)] = R
                self._mobile_com[label][(seg, chain, resnum)] = m_com
                self._target_com[label][(seg, chain, resnum)] = t_com
                df_ = df.copy()
                df_[['c_x', 'c_y', 'c_z']] = np.dot(df_[['c_x', 'c_y', 'c_z']] - m_com, R) + t_com
                self._ideal_ala_df[label][(seg, chain, resnum)] = df_

    def _import_sig_reps(self):
        labels_resns = defaultdict(set)
        for tup in self.sequence_csts.keys():
            for label in self.sequence_csts[tup].keys():
                labels_resns[label] |= set(self.sequence_csts[tup][label])
        for label in labels_resns.keys():
            for resn in labels_resns[label]:
                reppath = self.path + label + '/' + resn + self.filetype
                try:
                    if self.filetype == '.feather':
                        # print(label, resn)
                        self._sig_reps[label][resn] = pd.read_feather(reppath)
                    elif self.filetype == '.pkl':
                        with open(reppath, 'rb') as infile:
                            self._sig_reps[label][resn] = pickle.load(infile)
                except FileNotFoundError:
                    # print('hmm')
                    pass

    # def _import_additional_sig_rep(self, path, label, resn):
    #     try:
    #         with open(path + label + '/' + resn + '.pkl', 'rb') as infile:
    #             self._sig_reps[label][resn] = pickle.load(infile)
    #     except FileNotFoundError:
    #         pass

    @staticmethod
    def _get_phi_psi_df(df, phi, psi, phipsi_width=60):
        if phi is not None:
            phi_high = df['phi'] < (phi + (phipsi_width / 2))
            phi_low = df['phi'] > (phi - (phipsi_width / 2))
        else:
            phi_high = np.array([True] * len(df))
            phi_low = phi_high
        if psi is not None:
            psi_high = df['psi'] < (psi + (phipsi_width / 2))
            psi_low = df['psi'] > (psi - (phipsi_width / 2))
        else:
            psi_high = np.array([True] * len(df))
            psi_low = psi_high
        return df[phi_high & phi_low & psi_high & psi_low]

    @staticmethod
    def chunk_df(df_gr, gr_chunk_size=100):
        grs = list()
        for i, (n, gr) in enumerate(df_gr):
            grs.append(gr)
            if (i + 1) % gr_chunk_size == 0:
                yield pd.concat(grs)
                grs = list()

    def _load(self, template, seg, chain, resnum, **kwargs):
        # dataframe = pd.DataFrame()
        # t0 = time.time()
        phipsi_width = kwargs.get('phipsi_width', 60)

        dfs = list()
        for label in self.sequence_csts[(seg, chain, resnum)].keys():
            print('loading ' + str((seg, chain, resnum)) + ' , ' + label)
            if label == 'PHI_PSI':
                df_list = list()
                phi, psi = template.get_phi_psi(seg, chain, resnum)
                for resn in self.sequence_csts[(seg, chain, resnum)][label]:
                    df_phipsi = self._get_phi_psi_df(self._sig_reps[label][resn],
                                                     phi, psi, phipsi_width)
                    df_list.append(df_phipsi)
                df = pd.concat(df_list)
            else:
                df = pd.concat([self._sig_reps[label][resn]
                                for resn in self.sequence_csts[(seg, chain, resnum)][label]])

            if self.remove_from_df is not None:
                for d in self.remove_from_df.values():
                    tests = []
                    for col, val in d.items():
                        tests.append(df[col] == val)
                    tests = np.array(tests).T
                    tests = tests.all(axis=1)
                    df = df.loc[~tests]

            m_com = self._mobile_com[label][(seg, chain, resnum)]
            t_com = self._target_com[label][(seg, chain, resnum)]
            R = self._rot[label][(seg, chain, resnum)]
            print('transforming coordinates...')
            # df['coords'] = df['coords'].transform(lambda x: np.dot((x - m_com), R) + t_com)
            # df['coords'] = rottrans(np.stack(df.coords), m_com, R, t_com)
            # df['coords'] = list(np.dot(np.stack(df.coords) - m_com, R) + t_com)
            # df[['c_x', 'c_y', 'c_z']] = np.dot(df[['c_x', 'c_y', 'c_z']] - m_com, R) + t_com
            df[coords[:3]] = np.dot(df[coords[:3]] - m_com, R) + t_com
            df[coords[3:6]] = np.dot(df[coords[3:6]] - m_com, R) + t_com
            df[coords[6:9]] = np.dot(df[coords[6:9]] - m_com, R) + t_com
            df[coords[9:12]] = np.dot(df[coords[9:12]] - m_com, R) + t_com
            df[coords[12:15]] = np.dot(df[coords[12:15]] - m_com, R) + t_com
            df[coords[15:18]] = np.dot(df[coords[15:18]] - m_com, R) + t_com
            df[coords[18:21]] = np.dot(df[coords[18:21]] - m_com, R) + t_com
            df[coords[21:]] = np.dot(df[coords[21:]] - m_com, R) + t_com
            # df.loc[df['vec_acc_x'].notna(), vecsacc] = np.dot(df[vecsacc].dropna().values, R)
            # df.loc[df['vec_don1_x'].notna(), vecsd1] = np.dot(df[vecsd1].dropna().values, R)
            # df.loc[df['vec_don2_x'].notna(), vecsd2] = np.dot(df[vecsd2].dropna().values, R)
            # df.loc[df['vec_don3_x'].notna(), vecsd3] = np.dot(df[vecsd3].dropna().values, R)
            df['seg_chain_resnum'] = [(seg, chain, resnum)] * len(df)
            df['seg_chain_resnum_'] = [seg + '_' + chain + '_' + str(resnum)] * len(df)
            # df['seg_chain_resnum'] = [seg + '_' + chain + '_' + str(resnum)] * len(df)
            ###NEW STUFF FOR CLASH FILTER TEST
            df['str_index'] = df['iFG_count'] + '_' + df['vdM_count'] + '_' + df['query_name'] + '_' + df[
                'seg_chain_resnum_']
            # df['hash'] = df['str_index'].apply(hash)
            print('making transformed dataframe...')
            dfs.append(df)
            # dataframe = pd.concat((dataframe, df), sort=False)
        dataframe = pd.concat(dfs, sort=False, ignore_index=True)

        print('removing clashes...')
        df_nonclash = self._remove(dataframe, template, seg, chain, resnum, **kwargs)
        self._nonclashing.append(df_nonclash)
        # self._nonclashing.append(dataframe)  ##CHANGE previous 3 lines BACK!!
        # tf = time.time()
        # print('loaded (' + seg + ', ' + chain + ', ' + str(resnum) + ') in ' + str(tf-t0) + ' seconds.')
        # print('concatenating non-clashing to dataframe')
        # self.dataframe = pd.concat((self.dataframe, df_nonclash))
        # self._set_grouped_dataframe()

        ###### IF WANT TO CHUNK DATAFRAME, UNCOMMENT below code and COMMENT above block ###
        # print('removing clashes...')
        # dataframe_gr = dataframe.groupby(['iFG_count', 'vdM_count',
        #                                   'query_name', 'seg_chain_resnum'])
        #
        # for df_chnk in self.chunk_df(dataframe_gr, gr_chunk_size=500):
        #     df_nonclash = self._remove(df_chnk, template, seg, chain, resnum, **kwargs)
        #     print('concatenating non-clashing chunk to dataframe')
        #     self.dataframe = pd.concat((self.dataframe, df_nonclash), ignore_index=True)
        # self._set_grouped_dataframe()

    # @staticmethod
    # def _remove(dataframe, template, seg, chain, resnum, **kwargs):
    #     cla = Clash(df=template.dataframe, df_query=dataframe,
    #                 exclude=(resnum, chain, seg), **kwargs)
    #     cla.find()
    #     return cla.nonclashing_vdms
    @staticmethod
    def _remove(dataframe, template, seg, chain, resnum, **kwargs):
        t0 = time.time()
        cla = ClashVDM(dfq=dataframe, dft=template.dataframe)
        # cla.set_grouping(['iFG_count', 'vdM_count',
        #                   'query_name', 'seg_chain_resnum'])
        # cla.set_grouping(['iFG_count', 'vdM_count', 'query_name'])  # commented out for clashfilter test
        cla.set_grouping('str_index')
        cla.set_exclude((resnum, chain, seg))
        cla.setup()
        cla.find(**kwargs)
        tf = time.time()
        print('time:', tf - t0)
        return cla.dfq_clash_free

    def load(self, template, **kwargs):
        if not self._sig_reps:
            self._import_sig_reps()

        if not self._rot:
            self.set_rot_trans(template)

        for seg, chain, resnum in self.sequence_csts.keys():
            self._load(template, seg, chain, resnum, **kwargs)

        print('concatenating non-clashing to dataframe')
        t0 = time.time()
        self.dataframe = pd.concat(self._nonclashing, sort=False, ignore_index=True)
        self._nonclashing = list()
        self._sig_reps = defaultdict(dict)
        tf = time.time() - t0
        print('concatenated in ' + str(tf) + ' seconds.')
        self._set_grouped_dataframe()

    def load_additional(self, template, sequence_csts, **kwargs):
        seq_csts = defaultdict(dict)
        seq_csts_copy = copy.deepcopy(self.sequence_csts)
        for seg_ch_rn in sequence_csts.keys():
            if seg_ch_rn not in self.sequence_csts.keys():
                seq_csts[seg_ch_rn] = sequence_csts[seg_ch_rn]
                seq_csts_copy[seg_ch_rn] = sequence_csts[seg_ch_rn]

        if len(seq_csts.keys()) > 0:
            self.path = kwargs.get('path', self.path)
            self.sequence_csts = seq_csts
            self._import_sig_reps()
            self.set_rot_trans(template)
            self._nonclashing = list()

            for seg, chain, resnum in self.sequence_csts.keys():
                self._load(template, seg, chain, resnum, **kwargs)

            print('concatenating non-clashing to dataframe')
            t0 = time.time()
            _dataframe = pd.concat(self._nonclashing, sort=False, ignore_index=True)
            self.dataframe = pd.concat((self.dataframe, _dataframe), sort=False, ignore_index=True)
            self._nonclashing = list()
            self._sig_reps = defaultdict(dict)
            tf = time.time() - t0
            print('concatenated in ' + str(tf) + ' seconds.')
            self._set_grouped_dataframe()
            self.sequence_csts = seq_csts_copy
            return True
        else:
            return False

    def _set_grouped_dataframe(self):
        # self.dataframe_grouped = self.dataframe.groupby(['iFG_count', 'vdM_count',
        #                                                  'query_name', 'seg_chain_resnum'])
        self.dataframe_grouped = self.dataframe.groupby('str_index')


class VdM(Load):
    """Doesn't yet deal with terminal residues (although phi/psi does)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = kwargs.get('name', 'iFG_type')
        self.neighbors = None
        self.num_iFG_atoms = 0
        self.dataframe_iFG_coords = None
        self.neighbors = None
        self.ligand_iFG_corr_sorted = None
        self.path_to_pdbs = kwargs.get('path_to_pdbs', './')  # for printing vdMs
        if self.path_to_pdbs[-1] != '/':
            self.path_to_pdbs += '/'
        self.ligand_iFG_corr = kwargs.get('ligand_iFG_corr')  # use make_df_corr for dataframe

    # def _get_ifg_coords(self, df):
    #     df_ifg = df[df.chain == 'Y']
    #     # df_ifg_corr = pd.merge(self.ligand_iFG_corr[['resname', 'name']],
    #     #                        df_ifg, on=['resname', 'name'])
    #     df_ifg_corr = pd.merge(self.ligand_iFG_corr_sorted[['resname', 'name']],
    #                            df_ifg, on=['resname', 'name'])
    #     return np.stack(df_ifg_corr['coords']).reshape(-1)

    def _get_num_iFG_atoms(self):
        d = defaultdict(set)
        for n1, g1 in self.ligand_iFG_corr.groupby('lig_resname'):
            for n2, g2 in g1.groupby('resname'):
                d[n1] |= {len(g2)}
        for key in d.keys():
            assert len(d[key]) == 1, 'Problem with ligand iFG correspondence?'
            self.num_iFG_atoms += d[key].pop()

    def set_sorted_lig_corr(self):
        self.ligand_iFG_corr_sorted = self.ligand_iFG_corr.sort_values(by=['lig_resname', 'lig_name'])

    def set_neighbors(self, rmsd=0.4):
        if self.ligand_iFG_corr_sorted is None:
            self.set_sorted_lig_corr()

        if self.num_iFG_atoms == 0:
            self._get_num_iFG_atoms()

        # DONT NEED GROUPED BECAUSE OF INDEXING TRICK BELOW
        # if not self.dataframe_grouped:
        #     self._set_grouped_dataframe()

        df_ifg = self.dataframe[self.dataframe['chain'] == 'Y']
        df_ifg = pd.merge(self.ligand_iFG_corr_sorted[['resname', 'name']], df_ifg,
                          how='inner', on=['resname', 'name'], sort=False)
        # df_ifg = df_ifg.sort_values(by=['resname','name'])
        # df_ifg.set_index(['iFG_count', 'vdM_count', 'query_name', 'seg_chain_resnum'], inplace=True)
        # df_ifg_gr = df_ifg[['c_x', 'c_y', 'c_z']].groupby(df_ifg.index, sort=False)
        # self.dataframe_iFG_coords = df_ifg_gr.apply(lambda x: x.values.flatten())
        M = int(len(df_ifg) / self.num_iFG_atoms)
        N = self.num_iFG_atoms
        R = np.arange(len(df_ifg))
        # print('M', M)
        # print('N', N)
        # print('len(df_ifg)', len(df_ifg))
        # # self.dataframe_iFG_coords = np.stack(df_ifg.coords).reshape(M, N)
        # self.neighbors = NearestNeighbors(radius=np.sqrt(self.num_iFG_atoms) * rmsd, algorithm='ball_tree')
        # self.neighbors.fit(np.stack(self.dataframe_iFG_coords))
        inds = np.array([R[i::M] for i in range(M)]).flatten()
        # print('inds.shape', inds.shape)
        # self.dataframe_iFG_coords = df_ifg[:M][['iFG_count', 'vdM_count',
        #                                         'query_name', 'seg_chain_resnum']]
        self.dataframe_iFG_coords = df_ifg[:M]['str_index']
        self.neighbors = NearestNeighbors(radius=np.sqrt(self.num_iFG_atoms) * rmsd, algorithm='ball_tree')
        self.neighbors.fit(df_ifg.iloc[inds][['c_x', 'c_y', 'c_z']].values.reshape(M, N * 3))

    def _print_vdm(self, group_name, df_group, outpath, out_name_tag='', full_fragments=False,
                   with_bb=False, bfactor_field=None):
        """group_name is (ifg_count, vdm_count, query_name, seg_chain_resnum)"""
        if not full_fragments:
            if not with_bb:
                ProdyAG(**dict(bfactor_field=bfactor_field)).print_ag(group_name, df_group, outpath, out_name_tag)
            elif with_bb:
                label = set(df_group.label).pop()
                bb_names = rel_coords_dict[label]
                seg_chain_resnum = set(df_group.seg_chain_resnum).pop()
                df_ala = self._ideal_ala_df[label][seg_chain_resnum].copy()
                df_ala['segment'] = set(df_group.segment).pop()
                df_ala['resname'] = set(df_group.resname_vdm).pop()
                df_ala_bbsel = df_ala[df_ala['name'].isin(bb_names)]
                df = pd.concat((df_group, df_ala_bbsel))
                pdag = ProdyAG(**dict(bfactor_field=bfactor_field))
                pdag.print_ag(group_name, df, outpath, out_name_tag='_' + label + out_name_tag)
        else:
            pass  # update to print full fragments

    def print_vdms(self):
        pass