classdef design_parameters
    properties
        property_coupling
        design_number
        design_style
        design_options
        N_pix
    end
    methods
        function obj = design_parameters(design_number)
            if nargin == 1
                obj.property_coupling = 'coupled';
                obj.design_number = design_number;
                obj.design_style = 'matern52';
                obj.design_options = struct('sigma_f',1,'sigma_l',.5,'symmetry','none','N_value',3);
                obj.N_pix = [5 5];
            end
        end
        function obj = prepare(obj)
            obj = obj.expand_property_information;
        end
        function obj = expand_property_information(obj)
            % design_number
            if numel(obj.design_number) == 1
                obj.design_number = repmat(obj.design_number,1,3);
            end
            % design_style
            if isa(obj.design_style,'char')
                temp = obj.design_style;
                obj.design_style = cell(1,3);
                for i = 1:3
                    obj.design_style{i} = temp;
                end
            elseif isa(obj.design_style,'cell')
                if numel(obj.design_style) == 1
                    for i = 2:3
                        obj.design_style{i} = obj.design_style{1};
                    end
                end
            end
            % design_options
            if numel(obj.design_options) == 1
                temp = obj.design_options;
                obj.design_options = cell(1,3);
                [obj.design_options{:}] = deal(temp);
            end
        end
    end
end