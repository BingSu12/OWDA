function ct = centroid_init_avealign(sequences, template_length)

    seqnum = size(sequences,2); 
    alignpath = cell(1,seqnum);
    
    for i = 1:seqnum
        features = sequences{i};
        feanum = size(features,1);
        partemp_align_path = zeros(template_length,2);
        if feanum>=template_length
            partemp_ave_align_num = [floor(feanum/template_length) floor(feanum/template_length)];
            temp_start = 1;
            for temp_align_count = 1:template_length-1
                temp_end = temp_start + partemp_ave_align_num(mod(temp_align_count,2)+1)-1;
                partemp_align_path(temp_align_count,:) = [temp_start temp_end];
                temp_start = temp_end + 1;
            end
            temp_end = feanum;
            partemp_align_path(template_length,:) = [temp_start temp_end];
        else
            partemp_ave_align_num = [max(floor(template_length/feanum),1) max(floor(template_length/feanum),1)];
            temp_start = 1;
            for temp_align_count = 1:feanum-1
                temp_end = temp_start + partemp_ave_align_num(mod(temp_align_count,2)+1)-1;
                for temp_tem_count = temp_start:temp_end
                    partemp_align_path(temp_tem_count,:) = [temp_align_count temp_align_count];
                end
                temp_start = temp_end + 1;
            end
            temp_end = template_length;
            temp_align_count = feanum;
            for temp_tem_count = temp_start:temp_end
                partemp_align_path(temp_tem_count,:) = [temp_align_count temp_align_count];
            end
        end
        alignpath{i} = partemp_align_path;
    end
    
    dim = size(features,2);
    mean_sequence = zeros(template_length,dim);
    mean_sequence_num = zeros(1, template_length);
    for i = 1:seqnum
        features = sequences{i};
        temp_align_path = alignpath{i};
        for temp_align_count = 1:template_length
            temp_start = temp_align_path(temp_align_count,1);
            temp_end = temp_align_path(temp_align_count,2);
            mean_sequence_num(temp_align_count) = mean_sequence_num(temp_align_count) + temp_end - temp_start + 1;
            mean_sequence(temp_align_count,:) = mean_sequence(temp_align_count,:) + sum(features([temp_start:temp_end],:),1);             
        end           
    end
    for temp_align_count = 1:template_length
        if mean_sequence_num(temp_align_count) > 0
            mean_sequence(temp_align_count,:) = mean_sequence(temp_align_count,:)./mean_sequence_num(temp_align_count);
        end
    end
    
    ct.supp = mean_sequence';
    support_size=size(ct.supp,2);
    ct.w = ones(1,support_size)/support_size;   
end